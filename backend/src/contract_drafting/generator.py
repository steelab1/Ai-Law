"""
Contract Generator
Core logic for generating contracts from templates
"""

import logging
import re
import time
import uuid
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import (
    get_template_by_id,
    save_draft,
    get_draft_by_id,
    get_user_drafts,
    count_user_drafts
)
from .prompts import (
    CONTRACT_DRAFTING_SYSTEM,
    ENHANCE_CONTRACT_PROMPT,
    RAG_QUERIES,
    LEGAL_REFERENCES
)
from .schemas import (
    ContractType,
    ContractDraftRequest,
    ContractDraftResponse,
    TemplateListResponse,
    TemplateListItem,
    TemplateDetailResponse,
    Placeholder,
    DraftHistoryResponse,
    DraftHistoryItem
)

logger = logging.getLogger(__name__)


class ContractGenerator:
    """Generate contracts from templates with LLM enhancement."""

    def __init__(self, retrieval_url: str = "http://localhost:8002"):
        """
        Initialize generator.

        Args:
            retrieval_url: Base URL for retrieval API
        """
        self.retrieval_url = retrieval_url

    def _fill_template(self, template: Dict[str, Any], values: Dict[str, Any]) -> str:
        """
        Fill template placeholders with provided values.

        Args:
            template: Template document
            values: Placeholder values

        Returns:
            Filled contract text
        """
        contract_parts = []

        for section in sorted(template.get("sections", []), key=lambda x: x.get("order", 0)):
            content = section.get("content_template", "")

            # Replace placeholders
            for placeholder in section.get("placeholders", []):
                key = placeholder["key"]
                pattern = r"\{\{" + key + r"\}\}"

                if key in values:
                    value = str(values[key])
                elif placeholder.get("default_value"):
                    value = placeholder["default_value"]
                elif placeholder.get("required", True):
                    value = f"[{placeholder.get('label', key)}]"  # Keep as placeholder
                else:
                    value = ""

                content = re.sub(pattern, value, content)

            if content.strip():
                contract_parts.append(content)

        return "\n\n".join(contract_parts)

    async def _retrieve_legal_docs(
        self,
        contract_type: str,
        additional_queries: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve related legal documents via RAG.

        Args:
            contract_type: Type of contract
            additional_queries: Extra queries to search

        Returns:
            Formatted legal context string
        """
        queries = RAG_QUERIES.get(contract_type, [])
        if additional_queries:
            queries.extend(additional_queries)

        if not queries:
            return ""

        all_docs = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries[:3]:  # Limit to avoid too many requests
                try:
                    response = await client.post(
                        f"{self.retrieval_url}/retrieval",
                        json={"message": query, "top_k": 3}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        docs = data.get("documents", [])
                        all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to retrieve docs for '{query}': {e}")

        # Deduplicate and format
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_text = doc if isinstance(doc, str) else doc.get("content", "")
            if doc_text and doc_text not in seen:
                seen.add(doc_text)
                unique_docs.append(doc_text)

        return "\n\n".join(unique_docs[:10])  # Limit context size

    async def _enhance_with_llm(
        self,
        draft: str,
        legal_context: str,
        additional_requirements: Optional[str] = None
    ) -> str:
        """
        Enhance contract draft with LLM.

        Args:
            draft: Initial contract draft
            legal_context: Retrieved legal documents
            additional_requirements: Extra user requirements

        Returns:
            Enhanced contract text
        """
        from brain import openai_chat_complete

        additional_req_text = ""
        if additional_requirements:
            additional_req_text = f"\nYeu cau bo sung tu nguoi dung: {additional_requirements}"

        prompt = ENHANCE_CONTRACT_PROMPT.format(
            contract_draft=draft,
            legal_context=legal_context or "Khong co tai lieu bo sung.",
            additional_requirements=additional_req_text
        )

        messages = [
            {"role": "system", "content": CONTRACT_DRAFTING_SYSTEM},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai_chat_complete(messages)
            return response
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return draft  # Return original if enhancement fails

    async def generate_contract(self, request: ContractDraftRequest) -> ContractDraftResponse:
        """
        Generate a contract from template.

        Args:
            request: Draft request with template ID and values

        Returns:
            Generated contract response
        """
        start_time = time.time()

        # Get template
        template = get_template_by_id(request.template_id)
        if not template:
            raise ValueError(f"Template not found: {request.template_id}")

        contract_type = template.get("template_type", "other")

        # Step 1: Fill template with values
        initial_draft = self._fill_template(template, request.values)

        # Step 2: Retrieve legal context (if enabled)
        legal_context = ""
        legal_refs = LEGAL_REFERENCES.get(contract_type, [])

        if request.include_legal_references:
            legal_context = await self._retrieve_legal_docs(contract_type)

        # Step 3: Enhance with LLM
        final_contract = await self._enhance_with_llm(
            initial_draft,
            legal_context,
            request.additional_clauses
        )

        # Generate warnings for unfilled placeholders
        warnings = []
        unfilled = re.findall(r'\[([^\]]+)\]', final_contract)
        if unfilled:
            warnings.append(f"Cac truong chua dien: {', '.join(set(unfilled))}")

        # Save draft
        draft_id = f"draft_{uuid.uuid4().hex[:12]}"
        processing_time = time.time() - start_time

        draft_doc = {
            "draft_id": draft_id,
            "user_id": request.user_id,
            "template_id": request.template_id,
            "template_name": template.get("template_name", ""),
            "contract_type": contract_type,
            "contract_content": final_contract,
            "input_values": request.values,
            "legal_references": legal_refs,
            "warnings": warnings,
            "processing_time_seconds": processing_time
        }

        save_draft(draft_doc)

        logger.info(f"Generated contract {draft_id} in {processing_time:.2f}s")

        return ContractDraftResponse(
            draft_id=draft_id,
            user_id=request.user_id,
            template_id=request.template_id,
            template_name=template.get("template_name", ""),
            contract_type=ContractType(contract_type),
            contract_content=final_contract,
            legal_references=legal_refs,
            warnings=warnings,
            created_at=datetime.utcnow(),
            processing_time_seconds=processing_time
        )

    def get_draft(self, draft_id: str) -> Optional[ContractDraftResponse]:
        """Get a saved draft by ID."""
        draft = get_draft_by_id(draft_id)
        if not draft:
            return None

        return ContractDraftResponse(
            draft_id=draft["draft_id"],
            user_id=draft["user_id"],
            template_id=draft["template_id"],
            template_name=draft.get("template_name", ""),
            contract_type=ContractType(draft.get("contract_type", "other")),
            contract_content=draft["contract_content"],
            legal_references=draft.get("legal_references", []),
            warnings=draft.get("warnings", []),
            created_at=draft.get("created_at", datetime.utcnow()),
            processing_time_seconds=draft.get("processing_time_seconds", 0)
        )

    def get_user_draft_history(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> DraftHistoryResponse:
        """Get draft history for a user."""
        drafts = get_user_drafts(user_id, limit, skip)
        total = count_user_drafts(user_id)

        items = [
            DraftHistoryItem(
                draft_id=d["draft_id"],
                template_name=d["template_name"],
                contract_type=ContractType(d.get("contract_type", "other")),
                created_at=d["created_at"],
                preview=d["preview"]
            )
            for d in drafts
        ]

        return DraftHistoryResponse(drafts=items, total=total)

    @staticmethod
    def get_all_placeholders(template: Dict[str, Any]) -> List[Placeholder]:
        """Extract all placeholders from template for form generation."""
        placeholders = []
        seen_keys = set()

        for section in template.get("sections", []):
            for ph in section.get("placeholders", []):
                if ph["key"] not in seen_keys:
                    seen_keys.add(ph["key"])
                    placeholders.append(Placeholder(
                        key=ph["key"],
                        label=ph.get("label", ph["key"]),
                        type=ph.get("type", "string"),
                        required=ph.get("required", True),
                        default_value=ph.get("default_value"),
                        hint=ph.get("hint")
                    ))

        return placeholders
