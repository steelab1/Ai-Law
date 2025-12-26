import asyncio
import logging
import os
from copy import copy

from celery import shared_task

from utils import setup_logging
from database import get_celery_app
from brain import detect_route, openai_chat_complete, detect_user_intent, gen_doc_prompt, \
    get_legal_agent_anwer
from agent import react_agent_handle
from models import update_chat_conversation, get_conversation_messages
import requests

setup_logging()
logger = logging.getLogger(__name__)

celery_app = get_celery_app(__name__)
celery_app.autodiscover_tasks()

# Read retrieval URL from environment
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://127.0.0.1:8002/retrieval")


@shared_task()
def bot_answer_message(history, message):
    user_intent = detect_user_intent(history, message)
    logger.info(f"User intent: {user_intent}")

    # Call api retrieval relevance document
    # With GPU: can use higher top_k for better results
    # Without GPU: reduce to 5 to limit CPU rerank time
    payload = {
        "query": user_intent,
        "top_k_search": 20,  # 20 per source, GPU rerank is fast
        "top_k_rerank": 5
    }
    headers = {
        "Content-Type": "application/json"
    }

    logger.info(f"Calling retrieval API: {RETRIEVAL_URL}")
    try:
        response = requests.post(RETRIEVAL_URL, json=payload, headers=headers, timeout=120)
    except requests.exceptions.Timeout:
        logger.error("Retrieval API timeout after 120s")
        top_docs = []
        response = None
    except requests.exceptions.RequestException as e:
        logger.error(f"Retrieval API error: {e}")
        top_docs = []
        response = None

    if response and response.status_code == 200:
        top_docs = response.json().get("results")
        logger.info(f"Retrieval returned {len(top_docs)} documents")
    elif response:
        logger.error(f"Retrieval API error: {response.status_code} - {response.text}")
        top_docs = []
    # else: top_docs already set in exception handlers

    # Use history as openai messages
    session_history = copy(history)

    openai_messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý pháp luật chuyên nghiệp, hãy trả lời câu hỏi của người dùng dựa trên lịch sử chat và các tài liệu pháp luật được cung cấp.

YÊU CẦU TRẢ LỜI:
- Trả lời chi tiết, đầy đủ với các điều khoản cụ thể từ văn bản pháp luật
- Trích dẫn rõ nguồn: số điều, khoản, tên văn bản pháp luật (Nghị định, Thông tư, Luật...)
- Nếu có mức phạt, nêu rõ mức tối thiểu và tối đa
- Giải thích ngắn gọn ý nghĩa thực tiễn nếu cần thiết
- Cấu trúc câu trả lời rõ ràng, dễ đọc

LƯU Ý:
- Chỉ trả lời dựa trên thông tin trong tài liệu được cung cấp
- Nếu không tìm thấy thông tin liên quan trong tài liệu, trả về đúng từ: "no"
- Không bịa đặt hoặc suy diễn thông tin ngoài tài liệu"""
            },
            *session_history
        ]

    # Update documents to prompt
    rag_openai_messages = openai_messages +  [
            {"role": "user", "content": gen_doc_prompt(top_docs)},
            {"role": "user", "content": message},
        ]

    assistant_answer = openai_chat_complete(rag_openai_messages)
    if assistant_answer != "no":
        logger.info(f"Only call RAG")
        return assistant_answer
    else:
        messages = history + [
            {"role": "user", "content": message},
        ]
        agent_answer = get_legal_agent_anwer(messages)
        return agent_answer


@shared_task()
def bot_route_answer_message(history, question):
    # detect the route
    route = detect_route(history, question)
    if route == "chitchat":
        logger.info(f"Router to chitchat")
        mess_format_openai = [
            {"role": "system", "content": "Là một trợ lý thông minh, hãy trả lời các câu hỏi này dựa theo tri thức của bạn và hãy trả về kết quả là tiếng Việt."},
            {"role": "user", "content": question}
            ]
        output_chitchat =  openai_chat_complete(mess_format_openai)
        return output_chitchat

    elif route == 'legal':
        logger.info("Router to legal topic")
        return bot_answer_message(history, question)

    else:
        return "Sorry, I don't understand your question."


@shared_task()
def llm_handle_message(bot_id, user_id, question):
    logger.info("Start handle message")
    # Update chat conversation for new user_question
    conversation_id = update_chat_conversation(bot_id, user_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[-5:-1]
    # Use bot route to handle message
    response = bot_route_answer_message(history, question)
    logger.info(f"Chatbot response: {response}")
    # Save response to history
    update_chat_conversation(bot_id, user_id, response, False)
    # Return response
    return {"role": "assistant", "content": response}
