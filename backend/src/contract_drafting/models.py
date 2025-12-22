"""
Contract Drafting Models
MongoDB operations for contract templates and drafts
"""

import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pymongo import MongoClient

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "final_project")

client = MongoClient(MONGODB_URL)
db = client[MONGODB_DATABASE]

# Collections
contract_templates = db["contract_templates"]
contract_drafts = db["contract_drafts"]


# --- Template Operations ---

def get_all_templates(
    template_type: Optional[str] = None,
    industry: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all contract templates with optional filtering.

    Args:
        template_type: Filter by contract type (labor, sales, etc.)
        industry: Filter by industry (general, real_estate, tech, etc.)

    Returns:
        List of template summaries
    """
    query = {}
    if template_type:
        query["template_type"] = template_type
    if industry:
        query["industry"] = industry

    templates = contract_templates.find(
        query,
        {
            "template_id": 1,
            "template_name": 1,
            "template_type": 1,
            "description": 1,
            "industry": 1,
            "sections": 1,
            "_id": 0
        }
    )

    result = []
    for t in templates:
        placeholder_count = sum(
            len(section.get("placeholders", []))
            for section in t.get("sections", [])
        )
        result.append({
            "template_id": t["template_id"],
            "template_name": t["template_name"],
            "template_type": t["template_type"],
            "description": t.get("description"),
            "industry": t.get("industry", "general"),
            "placeholder_count": placeholder_count
        })

    return result


def get_template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Get full template details by ID.

    Args:
        template_id: Template identifier

    Returns:
        Template document or None
    """
    template = contract_templates.find_one(
        {"template_id": template_id},
        {"_id": 0}
    )
    return template


def get_template_by_type(template_type: str) -> Optional[Dict[str, Any]]:
    """
    Get first template matching a type (for comparison).

    Args:
        template_type: Contract type (labor, sales, etc.)

    Returns:
        Template document or None
    """
    template = contract_templates.find_one(
        {"template_type": template_type},
        {"_id": 0}
    )
    return template


def save_template(template: Dict[str, Any]) -> str:
    """
    Save or update a contract template.

    Args:
        template: Template document

    Returns:
        Template ID
    """
    template["updated_at"] = datetime.utcnow()
    if "created_at" not in template:
        template["created_at"] = datetime.utcnow()

    contract_templates.update_one(
        {"template_id": template["template_id"]},
        {"$set": template},
        upsert=True
    )

    logger.info(f"Saved template: {template['template_id']}")
    return template["template_id"]


# --- Draft Operations ---

def save_draft(draft: Dict[str, Any]) -> str:
    """
    Save a generated contract draft.

    Args:
        draft: Draft document with contract content

    Returns:
        Draft ID
    """
    draft["created_at"] = datetime.utcnow()

    contract_drafts.insert_one(draft)

    logger.info(f"Saved draft: {draft['draft_id']} for user: {draft['user_id']}")
    return draft["draft_id"]


def get_draft_by_id(draft_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a draft by ID.

    Args:
        draft_id: Draft identifier

    Returns:
        Draft document or None
    """
    draft = contract_drafts.find_one(
        {"draft_id": draft_id},
        {"_id": 0}
    )
    return draft


def get_user_drafts(
    user_id: str,
    limit: int = 20,
    skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Get draft history for a user.

    Args:
        user_id: User identifier
        limit: Max results
        skip: Offset for pagination

    Returns:
        List of draft summaries
    """
    drafts = contract_drafts.find(
        {"user_id": user_id},
        {
            "draft_id": 1,
            "template_name": 1,
            "contract_type": 1,
            "created_at": 1,
            "contract_content": 1,
            "_id": 0
        }
    ).sort("created_at", -1).skip(skip).limit(limit)

    result = []
    for d in drafts:
        content = d.get("contract_content", "")
        result.append({
            "draft_id": d["draft_id"],
            "template_name": d["template_name"],
            "contract_type": d["contract_type"],
            "created_at": d["created_at"],
            "preview": content[:200] + "..." if len(content) > 200 else content
        })

    return result


def count_user_drafts(user_id: str) -> int:
    """Count total drafts for a user."""
    return contract_drafts.count_documents({"user_id": user_id})


def delete_draft(draft_id: str, user_id: str) -> bool:
    """
    Delete a draft (only by owner).

    Args:
        draft_id: Draft identifier
        user_id: Owner user ID

    Returns:
        True if deleted
    """
    result = contract_drafts.delete_one({
        "draft_id": draft_id,
        "user_id": user_id
    })
    return result.deleted_count > 0
