import logging
import time
from typing import Optional
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from utils import setup_logging
from tasks import llm_handle_message
from search_document.combine_search import CombinedSearch, SearchError
from api_docs import (
    APP_TITLE, APP_VERSION, APP_DESCRIPTION, APP_TAGS,
    RETRIEVAL_DOC, CHAT_COMPLETE_DOC, CHAT_COMPLETE_V2_DOC,
    LIST_TEMPLATES_DOC, TEMPLATE_DETAIL_DOC, CREATE_DRAFT_DOC, USER_DRAFTS_DOC,
    ANALYZE_CONTRACT_DOC, GET_ANALYSIS_DOC, USER_ANALYSES_DOC,
    COMPLETE_REQUEST_EXAMPLE, RETRIEVAL_REQUEST_EXAMPLE
)

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

# Init services
combined_search_instance = CombinedSearch()
contract_generator = ContractGenerator()
contract_comparator = ContractComparator()

# FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    openapi_tags=APP_TAGS
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class CompleteRequest(BaseModel):
    bot_id: Optional[str] = 'bot_Legal_VN'
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False

    class Config:
        json_schema_extra = {"example": COMPLETE_REQUEST_EXAMPLE}


class RetrievalRequest(BaseModel):
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5

    class Config:
        json_schema_extra = {"example": RETRIEVAL_REQUEST_EXAMPLE}


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/", tags=["Health Check"])
async def root():
    """Kiểm tra trạng thái server."""
    return {"message": "Hello World"}


# =============================================================================
# LEGAL Q&A ENDPOINTS
# =============================================================================

@app.post("/retrieval", tags=["Legal Q&A"])
async def retrieval(request: RetrievalRequest):
    __doc__ = RETRIEVAL_DOC
    try:
        start_total = time.time()

        query = request.query
        top_k_search = request.top_k_search
        top_k_rerank = request.top_k_rerank

        start_search = time.time()
        search_results = combined_search_instance.search(query_text=query, top_k=top_k_search)
        search_time = time.time() - start_search
        logger.info(f"[TIMING] Combined search: {search_time:.2f}s, found {len(search_results)} docs")

        results = search_results[:top_k_rerank] if len(search_results) > top_k_rerank else search_results

        total_time = time.time() - start_total
        logger.info(f"[TIMING] Total retrieval: {total_time:.2f}s")

        return {"results": results}

    except SearchError as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=503, detail="Search services unavailable")
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

retrieval.__doc__ = RETRIEVAL_DOC


@app.post("/chat/complete", tags=["Legal Q&A"])
async def complete(data: CompleteRequest):
    __doc__ = CHAT_COMPLETE_DOC
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

complete.__doc__ = CHAT_COMPLETE_DOC


@app.get("/chat/complete_v2/{task_id}", tags=["Legal Q&A"])
async def get_response(task_id: str):
    __doc__ = CHAT_COMPLETE_V2_DOC
    start_time = time.time()
    timeout = 60
    polling_interval = 0.1

    while True:
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        logger.info(f"Task ID: {task_id}, Status: {task_status}")

        if task_status not in ('PENDING', 'STARTED'):
            return {
                "task_id": task_id,
                "task_status": task_status,
                "task_result": task_result.result
            }

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
            return {
                "task_id": task_id,
                "task_status": task_status,
                "error_message": "Service timeout, please retry."
            }

        await asyncio.sleep(polling_interval)

get_response.__doc__ = CHAT_COMPLETE_V2_DOC


# =============================================================================
# CONTRACT DRAFTING ENDPOINTS
# =============================================================================

@app.get("/contract/templates", response_model=TemplateListResponse, tags=["Contract Drafting"])
async def list_templates(
    template_type: Optional[str] = None,
    industry: Optional[str] = None
):
    __doc__ = LIST_TEMPLATES_DOC
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

list_templates.__doc__ = LIST_TEMPLATES_DOC


@app.get("/contract/templates/{template_id}", response_model=TemplateDetailResponse, tags=["Contract Drafting"])
async def get_template_detail(template_id: str):
    __doc__ = TEMPLATE_DETAIL_DOC
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

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

get_template_detail.__doc__ = TEMPLATE_DETAIL_DOC


@app.post("/contract/draft", response_model=ContractDraftResponse, tags=["Contract Drafting"])
async def create_contract_draft(request: ContractDraftRequest):
    __doc__ = CREATE_DRAFT_DOC
    try:
        response = await contract_generator.generate_contract(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

create_contract_draft.__doc__ = CREATE_DRAFT_DOC


@app.get("/contract/drafts/{user_id}", response_model=DraftHistoryResponse, tags=["Contract Drafting"])
async def get_user_drafts(
    user_id: str,
    limit: int = 20,
    skip: int = 0
):
    __doc__ = USER_DRAFTS_DOC
    try:
        return contract_generator.get_user_draft_history(user_id, limit, skip)
    except Exception as e:
        logger.error(f"Error getting user drafts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

get_user_drafts.__doc__ = USER_DRAFTS_DOC


# =============================================================================
# CONTRACT ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/contract/analyze", response_model=ContractAnalysisResponse, tags=["Contract Analysis"])
async def analyze_contract(
    file: UploadFile = File(..., description="File hợp đồng (PDF, DOCX, DOC)"),
    user_id: str = Form(..., description="ID người dùng"),
    contract_type: Optional[str] = Form(None, description="Loại hợp đồng"),
    unfavorable_clauses: bool = Form(True, description="Phân tích điều khoản bất lợi"),
    standard_comparison: bool = Form(True, description="So sánh với mẫu chuẩn"),
    compliance_check: bool = Form(True, description="Kiểm tra tuân thủ pháp luật"),
    obligations_summary: bool = Form(True, description="Tóm tắt quyền và nghĩa vụ"),
    risk_assessment: bool = Form(True, description="Đánh giá rủi ro")
):
    __doc__ = ANALYZE_CONTRACT_DOC
    try:
        file_bytes = await file.read()

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

analyze_contract.__doc__ = ANALYZE_CONTRACT_DOC


@app.get("/contract/analysis/{analysis_id}", response_model=ContractAnalysisResponse, tags=["Contract Analysis"])
async def get_analysis(analysis_id: str, user_id: Optional[str] = None):
    __doc__ = GET_ANALYSIS_DOC
    result = contract_comparator.get_analysis(analysis_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result

get_analysis.__doc__ = GET_ANALYSIS_DOC


@app.get("/contract/analyses/{user_id}", response_model=AnalysisHistoryResponse, tags=["Contract Analysis"])
async def get_user_analyses(
    user_id: str,
    limit: int = 20,
    skip: int = 0
):
    __doc__ = USER_ANALYSES_DOC
    try:
        return contract_comparator.get_user_history(user_id, limit, skip)
    except Exception as e:
        logger.error(f"Error getting user analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

get_user_analyses.__doc__ = USER_ANALYSES_DOC


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        log_level="info",
        timeout_keep_alive=300
    )
