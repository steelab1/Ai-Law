"""
Contract Comparator
Core logic for analyzing and comparing contracts
"""

import asyncio
import json
import logging
import time
import uuid
import httpx
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .extractor import ContractExtractor
from .models import (
    save_analysis,
    get_analysis_by_id,
    get_analysis_by_hash,
    get_user_analyses,
    count_user_analyses
)
from .prompts import (
    CONTRACT_ANALYSIS_SYSTEM,
    CONTRACT_TYPE_DETECTION_PROMPT,
    UNFAVORABLE_CLAUSE_PROMPT,
    STANDARD_COMPARISON_PROMPT,
    COMPLIANCE_CHECK_PROMPT,
    OBLIGATIONS_SUMMARY_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    IMPROVEMENT_SUGGESTIONS_PROMPT,
    RAG_QUERIES
)
from .schemas import (
    ContractType,
    SeverityLevel,
    RiskLevel,
    ContractAnalysisRequest,
    ContractAnalysisResponse,
    ComparisonResult,
    UnfavorableClause,
    MissingClause,
    StandardComparison,
    ComplianceIssue,
    PartyObligation,
    RiskAssessment,
    RiskBreakdown,
    ImprovementSuggestion,
    AnalysisHistoryResponse,
    AnalysisHistoryItem
)

logger = logging.getLogger(__name__)

# Import template getter from drafting module
from contract_drafting.models import get_template_by_type


class ContractComparator:
    """Analyze and compare contracts against standards and legal requirements."""

    MAX_CHUNK_SIZE = 12000  # Characters per chunk for LLM

    def __init__(self, retrieval_url: str = "http://localhost:8002"):
        """
        Initialize comparator.

        Args:
            retrieval_url: Base URL for retrieval API
        """
        self.retrieval_url = retrieval_url

    def _call_llm(self, prompt: str, expect_json: bool = True) -> Dict[str, Any]:
        """
        Call LLM with prompt.

        Args:
            prompt: User prompt
            expect_json: Whether to parse response as JSON

        Returns:
            Parsed response or raw text
        """
        from brain import openai_chat_complete

        messages = [
            {"role": "system", "content": CONTRACT_ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai_chat_complete(messages)

            if expect_json:
                # Try to extract JSON from response
                try:
                    # Handle markdown code blocks
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0]
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0]

                    return json.loads(response.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}")
                    return {"raw_response": response}

            return {"raw_response": response}

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"error": str(e)}

    async def _retrieve_legal_docs(
        self,
        contract_type: str,
        additional_queries: Optional[List[str]] = None
    ) -> str:
        """
        Retrieve related legal documents via RAG.

        Args:
            contract_type: Type of contract
            additional_queries: Extra queries

        Returns:
            Formatted legal context
        """
        queries = RAG_QUERIES.get(contract_type, [])
        if additional_queries:
            queries.extend(additional_queries)

        if not queries:
            return ""

        all_docs = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries[:4]:
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

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_text = doc if isinstance(doc, str) else doc.get("content", "")
            if doc_text and doc_text not in seen:
                seen.add(doc_text)
                unique_docs.append(doc_text)

        return "\n\n".join(unique_docs[:8])

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into chunks."""
        if len(text) <= self.MAX_CHUNK_SIZE:
            return [text]

        chunks = []
        current_chunk = ""

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def detect_contract_type(self, contract_text: str) -> Tuple[ContractType, str]:
        """
        Detect contract type and generate summary.

        Returns:
            Tuple of (contract_type, summary)
        """
        # Use first chunk for detection
        text_sample = contract_text[:self.MAX_CHUNK_SIZE]

        prompt = CONTRACT_TYPE_DETECTION_PROMPT.format(contract_text=text_sample)
        result = self._call_llm(prompt)

        contract_type_str = result.get("contract_type", "other")
        try:
            contract_type = ContractType(contract_type_str)
        except ValueError:
            contract_type = ContractType.OTHER

        summary = result.get("contract_summary", "Khong the tao tom tat.")

        return contract_type, summary

    async def analyze_unfavorable_clauses(
        self,
        contract_text: str,
        contract_type: str,
        legal_context: str
    ) -> List[UnfavorableClause]:
        """Find unfavorable clauses in contract."""
        prompt = UNFAVORABLE_CLAUSE_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE],
            contract_type=contract_type,
            legal_context=legal_context or "Khong co tai lieu bo sung."
        )

        result = self._call_llm(prompt)
        clauses = []

        for item in result.get("unfavorable_clauses", []):
            try:
                severity = SeverityLevel(item.get("severity", "trung_binh"))
            except ValueError:
                severity = SeverityLevel.MEDIUM

            clauses.append(UnfavorableClause(
                clause_text=item.get("clause_text", ""),
                location=item.get("location"),
                reason=item.get("reason", ""),
                severity=severity,
                affected_party=item.get("affected_party"),
                suggestion=item.get("suggestion", ""),
                legal_reference=item.get("legal_reference")
            ))

        return clauses

    async def compare_with_standard(
        self,
        contract_text: str,
        contract_type: str,
        legal_context: str
    ) -> Optional[StandardComparison]:
        """Compare contract with standard template."""
        # Get standard template
        template = get_template_by_type(contract_type)
        if not template:
            logger.warning(f"No standard template for type: {contract_type}")
            return None

        # Build template text from sections
        template_text = "\n\n".join(
            section.get("content_template", "")
            for section in template.get("sections", [])
        )

        prompt = STANDARD_COMPARISON_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE],
            template_text=template_text[:self.MAX_CHUNK_SIZE // 2],
            contract_type=contract_type,
            legal_context=legal_context or ""
        )

        result = self._call_llm(prompt)

        missing_clauses = []
        for item in result.get("missing_mandatory_clauses", []):
            try:
                importance = SeverityLevel(item.get("importance", "trung_binh"))
            except ValueError:
                importance = SeverityLevel.MEDIUM

            missing_clauses.append(MissingClause(
                clause_name=item.get("clause_name", ""),
                importance=importance,
                suggested_text=item.get("suggested_text", ""),
                legal_requirement=item.get("legal_requirement")
            ))

        return StandardComparison(
            compared_against=template.get("template_name", "Mau chuan"),
            missing_mandatory_clauses=missing_clauses,
            deviations_from_standard=result.get("deviations_from_standard", []),
            additional_non_standard_clauses=result.get("additional_non_standard_clauses", []),
            conformity_score=float(result.get("conformity_score", 50))
        )

    async def check_compliance(
        self,
        contract_text: str,
        contract_type: str,
        legal_context: str
    ) -> List[ComplianceIssue]:
        """Check legal compliance."""
        prompt = COMPLIANCE_CHECK_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE],
            contract_type=contract_type,
            legal_context=legal_context or ""
        )

        result = self._call_llm(prompt)
        issues = []

        for item in result.get("compliance_issues", []):
            try:
                severity = SeverityLevel(item.get("severity", "trung_binh"))
            except ValueError:
                severity = SeverityLevel.MEDIUM

            issues.append(ComplianceIssue(
                issue=item.get("issue", ""),
                required_by=item.get("required_by", ""),
                legal_reference=item.get("legal_reference", ""),
                recommendation=item.get("recommendation", ""),
                severity=severity
            ))

        return issues

    async def summarize_obligations(self, contract_text: str) -> List[PartyObligation]:
        """Extract party obligations and rights."""
        prompt = OBLIGATIONS_SUMMARY_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE]
        )

        result = self._call_llm(prompt)
        obligations = []

        for item in result.get("party_obligations", []):
            obligations.append(PartyObligation(
                party_name=item.get("party_name", ""),
                obligations=item.get("obligations", []),
                rights=item.get("rights", []),
                balance_assessment=item.get("balance_assessment")
            ))

        return obligations

    async def assess_risk(
        self,
        contract_text: str,
        contract_type: str,
        unfavorable_count: int,
        compliance_count: int,
        conformity_score: float
    ) -> RiskAssessment:
        """Assess overall risk."""
        prompt = RISK_ASSESSMENT_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE // 2],
            contract_type=contract_type,
            unfavorable_count=unfavorable_count,
            compliance_count=compliance_count,
            conformity_score=conformity_score
        )

        result = self._call_llm(prompt)
        ra = result.get("risk_assessment", {})

        breakdown_data = ra.get("breakdown", {})
        breakdown = RiskBreakdown(
            legal_risk=float(breakdown_data.get("legal_risk", 5)),
            financial_risk=float(breakdown_data.get("financial_risk", 5)),
            operational_risk=float(breakdown_data.get("operational_risk", 5)),
            reputation_risk=float(breakdown_data.get("reputation_risk", 5))
        )

        try:
            risk_level = RiskLevel(ra.get("risk_level", "trung_binh"))
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        return RiskAssessment(
            overall_score=float(ra.get("overall_score", 5)),
            risk_level=risk_level,
            breakdown=breakdown,
            key_risks=ra.get("key_risks", []),
            summary=ra.get("summary", "Khong co danh gia.")
        )

    async def suggest_improvements(
        self,
        contract_text: str,
        contract_type: str,
        issues_summary: str
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions."""
        prompt = IMPROVEMENT_SUGGESTIONS_PROMPT.format(
            contract_text=contract_text[:self.MAX_CHUNK_SIZE // 2],
            contract_type=contract_type,
            issues_summary=issues_summary
        )

        result = self._call_llm(prompt)
        suggestions = []

        for item in result.get("improvement_suggestions", []):
            try:
                priority = SeverityLevel(item.get("priority", "trung_binh"))
            except ValueError:
                priority = SeverityLevel.MEDIUM

            suggestions.append(ImprovementSuggestion(
                section=item.get("section", ""),
                current_state=item.get("current_state"),
                suggestion=item.get("suggestion", ""),
                priority=priority,
                rationale=item.get("rationale", "")
            ))

        return suggestions

    async def analyze_contract(
        self,
        file_bytes: bytes,
        filename: str,
        request: ContractAnalysisRequest
    ) -> ContractAnalysisResponse:
        """
        Full contract analysis.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename
            request: Analysis request

        Returns:
            Complete analysis response
        """
        start_time = time.time()

        # Extract text
        contract_text, file_hash = ContractExtractor.extract(file_bytes, filename)

        # Check for existing analysis
        existing = get_analysis_by_hash(file_hash, request.user_id)
        if existing:
            logger.info(f"Found existing analysis for hash: {file_hash[:16]}")
            existing["analysis_id"] = existing.get("analysis_id", f"existing_{file_hash[:12]}")
            # Return existing but update ID if needed
            return ContractAnalysisResponse(
                analysis_id=existing["analysis_id"],
                user_id=existing["user_id"],
                filename=existing["filename"],
                file_hash=file_hash,
                extracted_text_preview=contract_text[:500],
                result=ComparisonResult(**existing["result"]),
                created_at=existing.get("created_at", datetime.utcnow()),
                processing_time_seconds=0.1
            )

        # Detect type
        if request.contract_type:
            contract_type = request.contract_type
            _, summary = await self.detect_contract_type(contract_text)
        else:
            contract_type, summary = await self.detect_contract_type(contract_text)

        contract_type_str = contract_type.value

        # Retrieve legal context
        legal_context = await self._retrieve_legal_docs(contract_type_str)

        # Run analyses in parallel based on options
        options = request.analysis_options
        tasks = []

        if options.get("unfavorable_clauses", True):
            tasks.append(("unfavorable", self.analyze_unfavorable_clauses(
                contract_text, contract_type_str, legal_context
            )))

        if options.get("standard_comparison", True):
            tasks.append(("standard", self.compare_with_standard(
                contract_text, contract_type_str, legal_context
            )))

        if options.get("compliance_check", True):
            tasks.append(("compliance", self.check_compliance(
                contract_text, contract_type_str, legal_context
            )))

        if options.get("obligations_summary", True):
            tasks.append(("obligations", self.summarize_obligations(contract_text)))

        # Execute parallel tasks
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[t[1] for t in tasks],
                return_exceptions=True
            )
            for (name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    logger.error(f"Task {name} failed: {result}")
                    results[name] = None
                else:
                    results[name] = result

        # Get counts for risk assessment
        unfavorable_clauses = results.get("unfavorable", []) or []
        compliance_issues = results.get("compliance", []) or []
        standard_comparison = results.get("standard")
        conformity_score = standard_comparison.conformity_score if standard_comparison else 50

        # Risk assessment (depends on previous results)
        risk_assessment = None
        if options.get("risk_assessment", True):
            risk_assessment = await self.assess_risk(
                contract_text,
                contract_type_str,
                len(unfavorable_clauses),
                len(compliance_issues),
                conformity_score
            )

        # Generate improvement suggestions
        issues_summary = f"""
        - Dieu khoan bat loi: {len(unfavorable_clauses)}
        - Van de tuan thu: {len(compliance_issues)}
        - Diem phu hop: {conformity_score}%
        """
        improvements = await self.suggest_improvements(
            contract_text, contract_type_str, issues_summary
        )

        # Build result
        comparison_result = ComparisonResult(
            detected_type=contract_type,
            contract_summary=summary,
            unfavorable_clauses=unfavorable_clauses,
            standard_comparison=standard_comparison,
            compliance_issues=compliance_issues,
            party_obligations=results.get("obligations", []) or [],
            risk_assessment=risk_assessment,
            improvement_suggestions=improvements,
            legal_references_used=[]  # Could extract from legal_context
        )

        # Save analysis
        analysis_id = f"analysis_{uuid.uuid4().hex[:12]}"
        processing_time = time.time() - start_time

        analysis_doc = {
            "analysis_id": analysis_id,
            "user_id": request.user_id,
            "filename": filename,
            "file_hash": file_hash,
            "extracted_text": contract_text,
            "result": comparison_result.model_dump(),
            "processing_time_seconds": processing_time
        }

        save_analysis(analysis_doc)

        logger.info(f"Completed analysis {analysis_id} in {processing_time:.2f}s")

        return ContractAnalysisResponse(
            analysis_id=analysis_id,
            user_id=request.user_id,
            filename=filename,
            file_hash=file_hash,
            extracted_text_preview=contract_text[:500],
            result=comparison_result,
            created_at=datetime.utcnow(),
            processing_time_seconds=processing_time
        )

    def get_analysis(self, analysis_id: str) -> Optional[ContractAnalysisResponse]:
        """Get saved analysis by ID."""
        analysis = get_analysis_by_id(analysis_id)
        if not analysis:
            return None

        return ContractAnalysisResponse(
            analysis_id=analysis["analysis_id"],
            user_id=analysis["user_id"],
            filename=analysis["filename"],
            file_hash=analysis.get("file_hash", ""),
            extracted_text_preview=analysis.get("extracted_text", "")[:500],
            result=ComparisonResult(**analysis["result"]),
            created_at=analysis.get("created_at", datetime.utcnow()),
            processing_time_seconds=analysis.get("processing_time_seconds", 0)
        )

    def get_user_history(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> AnalysisHistoryResponse:
        """Get analysis history for user."""
        analyses = get_user_analyses(user_id, limit, skip)
        total = count_user_analyses(user_id)

        items = []
        for a in analyses:
            try:
                detected_type = ContractType(a.get("detected_type", "other"))
            except ValueError:
                detected_type = ContractType.OTHER

            try:
                risk_level = RiskLevel(a.get("risk_level")) if a.get("risk_level") else None
            except ValueError:
                risk_level = None

            items.append(AnalysisHistoryItem(
                analysis_id=a["analysis_id"],
                filename=a["filename"],
                detected_type=detected_type,
                risk_level=risk_level,
                conformity_score=a.get("conformity_score"),
                issues_count=a.get("issues_count", 0),
                created_at=a["created_at"]
            ))

        return AnalysisHistoryResponse(analyses=items, total=total)
