"""
Contract Analysis Schemas
Pydantic models for contract comparison feature
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ContractType(str, Enum):
    """Supported contract types for analysis."""
    LABOR = "labor"
    SALES = "sales"
    RENTAL = "rental"
    SERVICES = "services"
    LOAN = "loan"
    PARTNERSHIP = "partnership"
    OTHER = "other"


class SeverityLevel(str, Enum):
    """Severity level for issues."""
    HIGH = "cao"
    MEDIUM = "trung_binh"
    LOW = "thap"


class RiskLevel(str, Enum):
    """Overall risk level."""
    HIGH = "cao"
    MEDIUM = "trung_binh"
    LOW = "thap"
    MINIMAL = "toi_thieu"


# --- Analysis Result Components ---

class UnfavorableClause(BaseModel):
    """An unfavorable or potentially harmful clause."""
    clause_text: str = Field(..., description="Original clause text")
    location: Optional[str] = Field(None, description="Location in document (e.g., Dieu 5)")
    reason: str = Field(..., description="Why this clause is unfavorable")
    severity: SeverityLevel
    affected_party: Optional[str] = Field(None, description="Which party is disadvantaged")
    suggestion: str = Field(..., description="Suggested modification")
    legal_reference: Optional[str] = Field(None, description="Related law if applicable")


class MissingClause(BaseModel):
    """A mandatory clause that is missing."""
    clause_name: str = Field(..., description="Name of missing clause")
    importance: SeverityLevel
    suggested_text: str = Field(..., description="Suggested clause text")
    legal_requirement: Optional[str] = Field(None, description="Law requiring this clause")


class ComplianceIssue(BaseModel):
    """A legal compliance issue."""
    issue: str = Field(..., description="Description of the issue")
    required_by: str = Field(..., description="What regulation requires this")
    legal_reference: str = Field(..., description="Specific law reference")
    recommendation: str = Field(..., description="How to fix the issue")
    severity: SeverityLevel


class PartyObligation(BaseModel):
    """Obligations and rights of a party."""
    party_name: str = Field(..., description="Party identifier (Ben A, Ben B, etc.)")
    obligations: List[str] = Field(default_factory=list, description="List of obligations")
    rights: List[str] = Field(default_factory=list, description="List of rights")
    balance_assessment: Optional[str] = Field(None, description="Assessment of rights/obligations balance")


class RiskBreakdown(BaseModel):
    """Detailed risk breakdown by category."""
    legal_risk: float = Field(..., ge=0, le=10, description="Legal risk score 0-10")
    financial_risk: float = Field(..., ge=0, le=10)
    operational_risk: float = Field(..., ge=0, le=10)
    reputation_risk: float = Field(..., ge=0, le=10)


class RiskAssessment(BaseModel):
    """Overall risk assessment."""
    overall_score: float = Field(..., ge=0, le=10, description="Overall risk score 0-10")
    risk_level: RiskLevel
    breakdown: RiskBreakdown
    key_risks: List[str] = Field(default_factory=list, description="Top risk factors")
    summary: str = Field(..., description="Risk assessment summary")


class StandardComparison(BaseModel):
    """Comparison with standard template."""
    compared_against: str = Field(..., description="Template used for comparison")
    missing_mandatory_clauses: List[MissingClause] = Field(default_factory=list)
    deviations_from_standard: List[str] = Field(default_factory=list)
    additional_non_standard_clauses: List[str] = Field(default_factory=list)
    conformity_score: float = Field(..., ge=0, le=100, description="Conformity percentage")


class ImprovementSuggestion(BaseModel):
    """A suggestion for improving the contract."""
    section: str = Field(..., description="Section to improve")
    current_state: Optional[str] = Field(None, description="Current text/state")
    suggestion: str = Field(..., description="Improvement suggestion")
    priority: SeverityLevel
    rationale: str = Field(..., description="Why this improvement is recommended")


# --- Main Result Model ---

class ComparisonResult(BaseModel):
    """Complete comparison/analysis result."""
    detected_type: ContractType
    contract_summary: str = Field(..., description="Brief summary of the contract")

    unfavorable_clauses: List[UnfavorableClause] = Field(default_factory=list)
    standard_comparison: Optional[StandardComparison] = None
    compliance_issues: List[ComplianceIssue] = Field(default_factory=list)
    party_obligations: List[PartyObligation] = Field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = None
    improvement_suggestions: List[ImprovementSuggestion] = Field(default_factory=list)

    legal_references_used: List[str] = Field(default_factory=list)


# --- Request/Response Models ---

class ContractAnalysisRequest(BaseModel):
    """Request to analyze an uploaded contract."""
    user_id: str
    contract_type: Optional[ContractType] = Field(
        None,
        description="If not provided, will auto-detect"
    )
    analysis_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "unfavorable_clauses": True,
            "standard_comparison": True,
            "compliance_check": True,
            "obligations_summary": True,
            "risk_assessment": True
        },
        description="Which analyses to perform"
    )


class ContractAnalysisResponse(BaseModel):
    """Full analysis response."""
    analysis_id: str
    user_id: str
    filename: str
    file_hash: str
    extracted_text_preview: str = Field(..., description="First 500 chars of extracted text")

    result: ComparisonResult

    created_at: datetime
    processing_time_seconds: float


class AnalysisHistoryItem(BaseModel):
    """Item in user's analysis history."""
    analysis_id: str
    filename: str
    detected_type: ContractType
    risk_level: Optional[RiskLevel]
    conformity_score: Optional[float]
    issues_count: int
    created_at: datetime


class AnalysisHistoryResponse(BaseModel):
    """Response for analysis history."""
    analyses: List[AnalysisHistoryItem]
    total: int
