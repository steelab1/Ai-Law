import logging
import time
from typing import Dict, Optional
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from utils import setup_logging
from tasks import llm_handle_message

from search_document.combine_search import CombinedSearch
# BGE-reranker đã bỏ để tiết kiệm ~5GB VRAM

# Contract modules
from contract_drafting.generator import ContractGenerator
from contract_drafting.models import get_all_templates, get_template_by_id
from contract_drafting.schemas import (
    ContractDraftRequest,
    ContractDraftResponse,
    TemplateListResponse,
    TemplateListItem,
    TemplateDetailResponse,
    DraftHistoryResponse
)
from contract_analysis.comparator import ContractComparator
from contract_analysis.schemas import (
    ContractAnalysisRequest,
    ContractAnalysisResponse,
    AnalysisHistoryResponse
)

setup_logging()
logger = logging.getLogger(__name__)

# init retriever (không dùng reranker để tiết kiệm VRAM)
combined_search_instance = CombinedSearch()

# init contract modules
contract_generator = ContractGenerator()
contract_comparator = ContractComparator()


app = FastAPI(
    title="Vietnamese Legal Q&A API",
    description="""
## Hệ thống Hỏi đáp Pháp luật Việt Nam

API hỗ trợ 3 chức năng chính:

### 1. Hỏi đáp pháp luật (Legal Q&A)
- Tìm kiếm văn bản pháp luật liên quan
- Trả lời câu hỏi về luật Việt Nam sử dụng RAG (Retrieval-Augmented Generation)

### 2. Soạn thảo hợp đồng (Contract Drafting)
- Lấy danh sách mẫu hợp đồng có sẵn
- Tạo hợp đồng từ mẫu với các thông tin tùy chỉnh
- Tham chiếu điều luật liên quan

### 3. Phân tích hợp đồng (Contract Analysis)
- Upload và phân tích hợp đồng từ file PDF/Word
- Phát hiện điều khoản bất lợi
- Kiểm tra tuân thủ pháp luật
- Đánh giá rủi ro

---
**Liên hệ:** support@example.com
    """,
    version="2.0.0",
    openapi_tags=[
        {
            "name": "Legal Q&A",
            "description": "API hỏi đáp và tìm kiếm văn bản pháp luật Việt Nam"
        },
        {
            "name": "Contract Drafting",
            "description": "API soạn thảo hợp đồng từ mẫu có sẵn"
        },
        {
            "name": "Contract Analysis",
            "description": "API phân tích và đánh giá rủi ro hợp đồng"
        }
    ]
)

# CORS middleware - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define class name
class CompleteRequest(BaseModel):
    """Schema cho request chat hoàn chỉnh"""
    bot_id: Optional[str] = 'bot_Legal_VN'
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False

    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "bot_Legal_VN",
                "user_id": "user_123456",
                "user_message": "Thời gian thử việc tối đa là bao lâu?",
                "sync_request": False
            }
        }

class RetrievalRequest(BaseModel):
    """Schema cho request tìm kiếm văn bản pháp luật"""
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Thời gian thử việc tối đa là bao lâu?",
                "top_k_search": 30,
                "top_k_rerank": 5
            }
        }


@app.get("/", tags=["Health Check"])
async def root():
    """
    Kiểm tra trạng thái server.

    Trả về message "Hello World" nếu server đang hoạt động.
    """
    return {"message": "Hello World"}

@app.post("/retrieval", tags=["Legal Q&A"])
async def retrieval(request: RetrievalRequest):
    """
    Tìm kiếm văn bản pháp luật liên quan đến câu hỏi.

    **Mô tả:**
    - Sử dụng kết hợp semantic search (paraphrase-vietnamese-law) và lexical search (Elasticsearch BM25)
    - Trả về danh sách các điều luật liên quan nhất

    **Tham số:**
    - **query**: Câu hỏi hoặc từ khóa tìm kiếm
    - **top_k_search**: Số lượng kết quả tìm kiếm ban đầu (mặc định: 30)
    - **top_k_rerank**: Số lượng kết quả trả về sau khi lọc (mặc định: 5)

    **Ví dụ câu hỏi:**
    - "Thời gian thử việc tối đa là bao lâu?"
    - "Mức phạt khi không đội mũ bảo hiểm?"
    - "Quy định về hợp đồng lao động"
    """
    try:
        import time as timing
        start_total = timing.time()

        # Lấy dữ liệu từ body
        query = request.query
        top_k_search = request.top_k_search
        top_k_rerank = request.top_k_rerank

        # Thực hiện tìm kiếm bằng CombinedSearch (legal embedding + elasticsearch)
        start_search = timing.time()
        search_results = combined_search_instance.search(query_text=query, top_k=top_k_search)
        search_time = timing.time() - start_search
        logger.info(f"[TIMING] Combined search: {search_time:.2f}s, found {len(search_results)} docs")

        # Trả về top_k_rerank kết quả (không rerank, để LLM tự chọn context)
        results = search_results[:top_k_rerank] if len(search_results) > top_k_rerank else search_results

        total_time = timing.time() - start_total
        logger.info(f"[TIMING] Total retrieval: {total_time:.2f}s")

        return {
            "results": results
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/chat/complete", tags=["Legal Q&A"])
async def complete(data: CompleteRequest):
    """
    Gửi câu hỏi và nhận câu trả lời từ AI về pháp luật Việt Nam.

    **Mô tả:**
    - Hệ thống sẽ phân loại câu hỏi (pháp luật / chitchat)
    - Nếu là câu hỏi pháp luật: tìm kiếm văn bản liên quan và sinh câu trả lời
    - Nếu không tìm thấy trong database: sử dụng Tavily web search

    **Chế độ xử lý:**
    - **sync_request = false** (mặc định): Trả về task_id, dùng API `/chat/complete_v2/{task_id}` để lấy kết quả
    - **sync_request = true**: Đợi và trả về kết quả ngay (có thể mất 10-30 giây)

    **Lưu ý:**
    - user_id được dùng để lưu lịch sử hội thoại
    - Mỗi user_id có context riêng biệt
    """
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(status_code=400, detail="User id and user message are required")

    if data.sync_request:
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete_v2/{task_id}", tags=["Legal Q&A"])
async def get_response(task_id: str):
    """
    Lấy kết quả trả lời từ task_id.

    **Mô tả:**
    - Endpoint này dùng để lấy kết quả sau khi gọi `/chat/complete` với sync_request=false
    - Hệ thống sẽ polling và đợi tối đa 60 giây

    **Trạng thái task (task_status):**
    - **PENDING**: Task đang chờ xử lý
    - **STARTED**: Task đang được xử lý
    - **SUCCESS**: Task hoàn thành, kết quả trong `task_result`
    - **FAILURE**: Task thất bại

    **Cách sử dụng:**
    1. Gọi POST `/chat/complete` → nhận `task_id`
    2. Gọi GET `/chat/complete_v2/{task_id}` → nhận kết quả
    """
    start_time = time.time()
    timeout = 60  # Timeout sau 60 giây
    polling_interval = 0.1  # Thời gian chờ giữa mỗi lần kiểm tra (100ms)
    
    while True:
        # Lấy trạng thái task từ Celery
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        
        # Ghi log trạng thái task
        logger.info(f"Task ID: {task_id}, Status: {task_status}")
        
        # Nếu task đã hoàn tất, trả về kết quả
        if task_status not in ('PENDING', 'STARTED'):
            return {
                "task_id": task_id,
                "task_status": task_status,
                "task_result": task_result.result
            }
        
        # Kiểm tra timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
            return {
                "task_id": task_id,
                "task_status": task_status,
                "error_message": "Service timeout, please retry."
            }
        
        # Chờ trước khi kiểm tra lại
        await asyncio.sleep(polling_interval)


# =============================================================================
# CONTRACT DRAFTING ENDPOINTS
# =============================================================================

@app.get("/contract/templates", response_model=TemplateListResponse, tags=["Contract Drafting"])
async def list_templates(
    template_type: Optional[str] = None,
    industry: Optional[str] = None
):
    """
    Lấy danh sách các mẫu hợp đồng có sẵn.

    **Mô tả:**
    - Trả về danh sách tất cả mẫu hợp đồng trong hệ thống
    - Có thể lọc theo loại hợp đồng hoặc ngành nghề

    **Loại hợp đồng (template_type):**
    - `labor`: Hợp đồng lao động
    - `sales`: Hợp đồng mua bán
    - `rental`: Hợp đồng thuê/cho thuê
    - `services`: Hợp đồng dịch vụ
    - `loan`: Hợp đồng vay/cho vay
    - `partnership`: Hợp đồng hợp tác

    **Ngành nghề (industry):**
    - `general`: Chung
    - `real_estate`: Bất động sản
    - `tech`: Công nghệ
    - `retail`: Bán lẻ
    """
    try:
        templates = get_all_templates(template_type, industry)
        items = [
            TemplateListItem(
                template_id=t["template_id"],
                template_name=t["template_name"],
                template_type=t["template_type"],
                description=t.get("description"),
                industry=t.get("industry", "general"),
                placeholder_count=t.get("placeholder_count", 0)
            )
            for t in templates
        ]
        return TemplateListResponse(templates=items, total=len(items))
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/templates/{template_id}", response_model=TemplateDetailResponse, tags=["Contract Drafting"])
async def get_template_detail(template_id: str):
    """
    Lấy chi tiết mẫu hợp đồng bao gồm các trường cần điền.

    **Mô tả:**
    - Trả về thông tin chi tiết của một mẫu hợp đồng
    - Bao gồm các section, điều khoản, và danh sách placeholder cần điền

    **Response bao gồm:**
    - `sections`: Các phần của hợp đồng
    - `legal_references`: Các điều luật tham chiếu
    - `all_placeholders`: Danh sách các trường cần điền giá trị
    """
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Extract all placeholders
    all_placeholders = ContractGenerator.get_all_placeholders(template)

    return TemplateDetailResponse(
        template_id=template["template_id"],
        template_name=template["template_name"],
        template_type=template["template_type"],
        description=template.get("description"),
        industry=template.get("industry", "general"),
        sections=template.get("sections", []),
        legal_references=template.get("legal_references", []),
        all_placeholders=all_placeholders
    )


@app.post("/contract/draft", response_model=ContractDraftResponse, tags=["Contract Drafting"])
async def create_contract_draft(request: ContractDraftRequest):
    """
    Tạo hợp đồng từ mẫu có sẵn.

    **Mô tả:**
    - Điền thông tin vào mẫu hợp đồng để tạo hợp đồng hoàn chỉnh
    - Có thể yêu cầu thêm điều khoản bổ sung
    - Tự động tham chiếu các điều luật liên quan

    **Tham số:**
    - **user_id**: ID người dùng (để lưu lịch sử)
    - **template_id**: ID mẫu hợp đồng (lấy từ API `/contract/templates`)
    - **values**: Object chứa giá trị các trường, ví dụ:
      ```json
      {
        "party_a_name": "Công ty TNHH ABC",
        "party_a_address": "123 Nguyễn Huệ, Q1, TP.HCM",
        "salary": "15,000,000 VND"
      }
      ```
    - **additional_clauses**: Yêu cầu bổ sung điều khoản (tùy chọn)
    - **include_legal_references**: Có lấy luật liên quan không (mặc định: true)
    """
    try:
        response = await contract_generator.generate_contract(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/drafts/{user_id}", response_model=DraftHistoryResponse, tags=["Contract Drafting"])
async def get_user_drafts(
    user_id: str,
    limit: int = 20,
    skip: int = 0
):
    """
    Lấy lịch sử các hợp đồng đã soạn của người dùng.

    **Mô tả:**
    - Trả về danh sách các hợp đồng đã được tạo bởi user
    - Hỗ trợ phân trang với `limit` và `skip`

    **Tham số:**
    - **user_id**: ID người dùng
    - **limit**: Số lượng kết quả tối đa (mặc định: 20)
    - **skip**: Bỏ qua bao nhiêu kết quả đầu tiên (mặc định: 0)

    **Ví dụ phân trang:**
    - Trang 1: `limit=20&skip=0`
    - Trang 2: `limit=20&skip=20`
    """
    try:
        return contract_generator.get_user_draft_history(user_id, limit, skip)
    except Exception as e:
        logger.error(f"Error getting user drafts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONTRACT ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/contract/analyze", response_model=ContractAnalysisResponse, tags=["Contract Analysis"])
async def analyze_contract(
    file: UploadFile = File(..., description="File hợp đồng (PDF, DOCX, DOC)"),
    user_id: str = Form(..., description="ID người dùng"),
    contract_type: Optional[str] = Form(None, description="Loại hợp đồng (tự động nhận diện nếu không cung cấp)"),
    unfavorable_clauses: bool = Form(True, description="Phân tích điều khoản bất lợi"),
    standard_comparison: bool = Form(True, description="So sánh với mẫu chuẩn"),
    compliance_check: bool = Form(True, description="Kiểm tra tuân thủ pháp luật"),
    obligations_summary: bool = Form(True, description="Tóm tắt quyền và nghĩa vụ"),
    risk_assessment: bool = Form(True, description="Đánh giá rủi ro")
):
    """
    Phân tích hợp đồng từ file PDF hoặc Word.

    **Mô tả:**
    - Upload file hợp đồng để AI phân tích
    - Hỗ trợ định dạng: PDF, DOCX, DOC
    - Tự động nhận diện loại hợp đồng nếu không chỉ định

    **Các tùy chọn phân tích:**
    - **unfavorable_clauses**: Phát hiện các điều khoản bất lợi cho bạn
    - **standard_comparison**: So sánh với mẫu hợp đồng chuẩn
    - **compliance_check**: Kiểm tra có vi phạm pháp luật không
    - **obligations_summary**: Tóm tắt quyền và nghĩa vụ các bên
    - **risk_assessment**: Đánh giá mức độ rủi ro (thang điểm 1-10)

    **Kết quả trả về:**
    - Điểm rủi ro và mức độ (low/medium/high)
    - Danh sách điều khoản bất lợi kèm gợi ý sửa đổi
    - Các vấn đề về tuân thủ pháp luật
    - Khuyến nghị cải thiện hợp đồng
    """
    try:
        # Read file content
        file_bytes = await file.read()

        # Build request
        from contract_analysis.schemas import ContractType as AnalysisContractType

        request = ContractAnalysisRequest(
            user_id=user_id,
            contract_type=AnalysisContractType(contract_type) if contract_type else None,
            analysis_options={
                "unfavorable_clauses": unfavorable_clauses,
                "standard_comparison": standard_comparison,
                "compliance_check": compliance_check,
                "obligations_summary": obligations_summary,
                "risk_assessment": risk_assessment
            }
        )

        response = await contract_comparator.analyze_contract(
            file_bytes=file_bytes,
            filename=file.filename,
            request=request
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/contract/analysis/{analysis_id}", response_model=ContractAnalysisResponse, tags=["Contract Analysis"])
async def get_analysis(analysis_id: str):
    """
    Lấy kết quả phân tích đã lưu.

    **Mô tả:**
    - Truy xuất kết quả phân tích hợp đồng từ database
    - Sử dụng `analysis_id` nhận được từ API `/contract/analyze`

    **Lưu ý:**
    - Trả về 404 nếu không tìm thấy kết quả
    """
    result = contract_comparator.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/contract/analyses/{user_id}", response_model=AnalysisHistoryResponse, tags=["Contract Analysis"])
async def get_user_analyses(
    user_id: str,
    limit: int = 20,
    skip: int = 0
):
    """
    Lấy lịch sử phân tích hợp đồng của người dùng.

    **Mô tả:**
    - Trả về danh sách các hợp đồng đã được phân tích bởi user
    - Hỗ trợ phân trang với `limit` và `skip`

    **Tham số:**
    - **user_id**: ID người dùng
    - **limit**: Số lượng kết quả tối đa (mặc định: 20)
    - **skip**: Bỏ qua bao nhiêu kết quả đầu tiên (mặc định: 0)

    **Response bao gồm:**
    - `analyses`: Danh sách các kết quả phân tích
    - `total`: Tổng số kết quả
    """
    try:
        return contract_comparator.get_user_history(user_id, limit, skip)
    except Exception as e:
        logger.error(f"Error getting user analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        log_level="info",
        timeout_keep_alive=300  # 5 minutes keep-alive timeout
    )

