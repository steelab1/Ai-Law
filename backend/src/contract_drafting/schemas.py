"""
Contract Drafting Schemas
Pydantic models for contract drafting feature
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ContractType(str, Enum):
    """Supported contract types."""
    LABOR = "labor"                 # Hop dong lao dong
    SALES = "sales"                 # Hop dong mua ban
    RENTAL = "rental"               # Hop dong thue
    SERVICES = "services"           # Hop dong dich vu
    LOAN = "loan"                   # Hop dong vay
    PARTNERSHIP = "partnership"     # Hop dong hop tac kinh doanh
    OTHER = "other"


class Placeholder(BaseModel):
    """Placeholder definition in template."""
    key: str = Field(..., description="Placeholder key, e.g., 'party_a_name'")
    label: str = Field(..., description="Display label in Vietnamese")
    type: str = Field(default="string", description="Data type: string, date, number, currency")
    required: bool = Field(default=True)
    default_value: Optional[str] = None
    hint: Optional[str] = Field(None, description="Helper text for user")


class TemplateSection(BaseModel):
    """A section within contract template."""
    section_id: str
    section_name: str = Field(..., description="Section name in Vietnamese")
    order: int = Field(..., description="Display order")
    content_template: str = Field(..., description="Template content with {{placeholders}}")
    placeholders: List[Placeholder] = Field(default_factory=list)
    is_optional: bool = Field(default=False)


class ContractTemplate(BaseModel):
    """Full contract template schema."""
    template_id: str
    template_type: ContractType
    template_name: str = Field(..., description="Template name in Vietnamese")
    description: Optional[str] = None
    industry: str = Field(default="general", description="Industry: general, real_estate, tech, etc.")
    sections: List[TemplateSection]
    rag_queries: List[str] = Field(
        default_factory=list,
        description="Queries for RAG to fetch related laws"
    )
    legal_references: List[str] = Field(
        default_factory=list,
        description="Primary legal references"
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# --- Request/Response Models ---

class ContractDraftRequest(BaseModel):
    """Request to draft a contract from template."""
    user_id: str
    template_id: str
    values: Dict[str, Any] = Field(
        ...,
        description="Placeholder values, e.g., {'party_a_name': 'Cong ty ABC'}"
    )
    additional_clauses: Optional[str] = Field(
        None,
        description="Additional requirements for LLM to include"
    )
    include_legal_references: bool = Field(
        default=True,
        description="Whether to fetch and include related laws"
    )


class ContractDraftResponse(BaseModel):
    """Response containing drafted contract."""
    draft_id: str
    user_id: str
    template_id: str
    template_name: str
    contract_type: ContractType
    contract_content: str = Field(..., description="Full contract text")
    legal_references: List[str] = Field(
        default_factory=list,
        description="Related laws used in drafting"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or suggestions"
    )
    created_at: datetime
    processing_time_seconds: float


class TemplateListItem(BaseModel):
    """Summary item for template listing."""
    template_id: str
    template_name: str
    template_type: ContractType
    description: Optional[str]
    industry: str
    placeholder_count: int


class TemplateListResponse(BaseModel):
    """Response for listing templates."""
    templates: List[TemplateListItem]
    total: int


class TemplateDetailResponse(BaseModel):
    """Detailed template with all placeholders."""
    template_id: str
    template_name: str
    template_type: ContractType
    description: Optional[str]
    industry: str
    sections: List[TemplateSection]
    legal_references: List[str]
    all_placeholders: List[Placeholder] = Field(
        ...,
        description="Flattened list of all placeholders for form generation"
    )


class DraftHistoryItem(BaseModel):
    """Item in user's draft history."""
    draft_id: str
    template_name: str
    contract_type: ContractType
    created_at: datetime
    preview: str = Field(..., description="First 200 chars of contract")


class DraftHistoryResponse(BaseModel):
    """Response for user's draft history."""
    drafts: List[DraftHistoryItem]
    total: int
