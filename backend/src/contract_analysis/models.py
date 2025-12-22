"""
Contract Analysis Models
MongoDB operations for contract analyses
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

# Collection
contract_analyses = db["contract_analyses"]


def save_analysis(analysis: Dict[str, Any]) -> str:
    """
    Save a contract analysis result.

    Args:
        analysis: Analysis document

    Returns:
        Analysis ID
    """
    analysis["created_at"] = datetime.utcnow()

    contract_analyses.insert_one(analysis)

    logger.info(f"Saved analysis: {analysis['analysis_id']} for user: {analysis['user_id']}")
    return analysis["analysis_id"]


def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Get analysis by ID.

    Args:
        analysis_id: Analysis identifier

    Returns:
        Analysis document or None
    """
    analysis = contract_analyses.find_one(
        {"analysis_id": analysis_id},
        {"_id": 0}
    )
    return analysis


def get_analysis_by_hash(file_hash: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get analysis by file hash (for deduplication).

    Args:
        file_hash: SHA256 hash of file
        user_id: User identifier

    Returns:
        Existing analysis or None
    """
    analysis = contract_analyses.find_one(
        {"file_hash": file_hash, "user_id": user_id},
        {"_id": 0}
    )
    return analysis


def get_user_analyses(
    user_id: str,
    limit: int = 20,
    skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Get analysis history for a user.

    Args:
        user_id: User identifier
        limit: Max results
        skip: Offset for pagination

    Returns:
        List of analysis summaries
    """
    analyses = contract_analyses.find(
        {"user_id": user_id},
        {
            "analysis_id": 1,
            "filename": 1,
            "result.detected_type": 1,
            "result.risk_assessment.risk_level": 1,
            "result.standard_comparison.conformity_score": 1,
            "result.unfavorable_clauses": 1,
            "result.compliance_issues": 1,
            "created_at": 1,
            "_id": 0
        }
    ).sort("created_at", -1).skip(skip).limit(limit)

    result = []
    for a in analyses:
        res = a.get("result", {})

        # Count issues
        issues_count = len(res.get("unfavorable_clauses", []))
        issues_count += len(res.get("compliance_issues", []))

        risk_assessment = res.get("risk_assessment", {})
        standard_comparison = res.get("standard_comparison", {})

        result.append({
            "analysis_id": a["analysis_id"],
            "filename": a["filename"],
            "detected_type": res.get("detected_type", "other"),
            "risk_level": risk_assessment.get("risk_level"),
            "conformity_score": standard_comparison.get("conformity_score"),
            "issues_count": issues_count,
            "created_at": a["created_at"]
        })

    return result


def count_user_analyses(user_id: str) -> int:
    """Count total analyses for a user."""
    return contract_analyses.count_documents({"user_id": user_id})


def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """
    Delete an analysis (only by owner).

    Args:
        analysis_id: Analysis identifier
        user_id: Owner user ID

    Returns:
        True if deleted
    """
    result = contract_analyses.delete_one({
        "analysis_id": analysis_id,
        "user_id": user_id
    })
    return result.deleted_count > 0


def update_analysis(analysis_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update an analysis.

    Args:
        analysis_id: Analysis identifier
        updates: Fields to update

    Returns:
        True if updated
    """
    updates["updated_at"] = datetime.utcnow()
    result = contract_analyses.update_one(
        {"analysis_id": analysis_id},
        {"$set": updates}
    )
    return result.modified_count > 0
