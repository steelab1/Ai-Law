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
from search_document.rerank import BGEReranker

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

# init retriever and reranker
combined_search_instance = CombinedSearch()
reranker_instance = BGEReranker(model_name="BAAI/bge-reranker-v2-m3", use_fp16=True)

# init contract modules
contract_generator = ContractGenerator()
contract_comparator = ContractComparator()


app = FastAPI(
    title="Vietnamese Legal Q&A API",
    description="API for legal Q&A, contract drafting, and contract analysis",
    version="2.0.0"
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
    bot_id: Optional[str] = 'bot_Legal_VN'
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False

class RetrievalRequest(BaseModel):
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/retrieval")
async def retrieval(request: RetrievalRequest):
    try:
        import time as timing
        start_total = timing.time()

        # Lấy dữ liệu từ body
        query = request.query
        top_k_search = request.top_k_search
        top_k_rerank = request.top_k_rerank

        # Thực hiện tìm kiếm bằng CombinedSearch
        start_search = timing.time()
        search_results = combined_search_instance.search(query_text=query, top_k=top_k_search)
        search_time = timing.time() - start_search
        logger.info(f"[TIMING] Combined search: {search_time:.2f}s, found {len(search_results)} docs")

        # Thực hiện rerank kết quả tìm kiếm
        start_rerank = timing.time()
        reranked_results = reranker_instance.rerank(query=query, documents=search_results, topk=top_k_rerank)
        rerank_time = timing.time() - start_rerank
        logger.info(f"[TIMING] Rerank {len(search_results)} docs: {rerank_time:.2f}s")

        total_time = timing.time() - start_total
        logger.info(f"[TIMING] Total retrieval: {total_time:.2f}s")

        return {
            "results": reranked_results
        }

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/chat/complete")
async def complete(data: CompleteRequest):
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


@app.get("/chat/complete_v2/{task_id}")
async def get_response(task_id: str):
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
    Lay danh sach cac mau hop dong co san.

    - **template_type**: Loc theo loai hop dong (labor, sales, rental, services, loan, partnership)
    - **industry**: Loc theo nganh (general, real_estate, tech, etc.)
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
    Lay chi tiet mau hop dong bao gom cac truong can dien.

    - **template_id**: ID cua mau hop dong
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
    Tao hop dong tu mau co san.

    - **user_id**: ID nguoi dung
    - **template_id**: ID mau hop dong
    - **values**: Gia tri cac truong (vd: {"party_a_name": "Cong ty ABC"})
    - **additional_clauses**: Yeu cau bo sung (optional)
    - **include_legal_references**: Co lay luat lien quan khong (default: true)
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
    Lay lich su cac hop dong da soan cua nguoi dung.

    - **user_id**: ID nguoi dung
    - **limit**: So luong ket qua toi da
    - **skip**: Bo qua bao nhieu ket qua (phan trang)
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
    file: UploadFile = File(...),
    user_id: str = Form(...),
    contract_type: Optional[str] = Form(None),
    unfavorable_clauses: bool = Form(True),
    standard_comparison: bool = Form(True),
    compliance_check: bool = Form(True),
    obligations_summary: bool = Form(True),
    risk_assessment: bool = Form(True)
):
    """
    Phan tich hop dong tu file PDF hoac Word.

    - **file**: File hop dong (PDF, DOCX, DOC)
    - **user_id**: ID nguoi dung
    - **contract_type**: Loai hop dong (tu dong nhan dien neu khong cung cap)
    - **unfavorable_clauses**: Phan tich dieu khoan bat loi
    - **standard_comparison**: So sanh voi mau chuan
    - **compliance_check**: Kiem tra tuan thu phap luat
    - **obligations_summary**: Tom tat quyen va nghia vu
    - **risk_assessment**: Danh gia rui ro
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
    Lay ket qua phan tich da luu.

    - **analysis_id**: ID ket qua phan tich
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
    Lay lich su phan tich hop dong cua nguoi dung.

    - **user_id**: ID nguoi dung
    - **limit**: So luong ket qua toi da
    - **skip**: Bo qua bao nhieu ket qua (phan trang)
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

