import os
import logging
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# Check if Tavily is enabled
TAVILY_ENABLED = os.getenv("TAVILY_ENABLED", "true").lower() == "true"

tavily_client = None
if TAVILY_ENABLED:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        logger.info("Tavily web search enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize Tavily: {e}")
        TAVILY_ENABLED = False
else:
    logger.info("Tavily web search disabled")


def search(query):
    """Search web using Tavily API"""
    if not TAVILY_ENABLED or tavily_client is None:
        logger.info("Tavily disabled, skipping web search")
        return "Web search is disabled."

    try:
        # get 3 first links
        output_search = tavily_client.search(query).get('results')[:3]
        # Xử lý kết quả tìm kiếm thành chuỗi tài liệu
        search_document = "Dưới đây là các tài liệu truy xuất được từ internet: \n"
        for i, doc in enumerate(output_search):
            search_document += f"{i+1}. {doc.get('content', '')} \n"
        search_document += "Kết thúc phần tài liệu truy xuất được."
        return search_document
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Web search failed: {e}"


def is_tavily_enabled():
    """Check if Tavily is enabled and working"""
    return TAVILY_ENABLED and tavily_client is not None


# define a function
functions_info = [
    {
        "name": "search",
        "description": "Get information in internet base on user query ",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "This is user query",
                },
            },
            "required": ["query"],
        },
    }
]
