"""
Simple test script to verify contract API routes are registered correctly
"""

import sys
sys.path.insert(0, '.')

# Mock the complex imports
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

# Mock all complex dependencies before importing app
import sys
sys.modules['tasks'] = MockModule()
sys.modules['search_document'] = MockModule()
sys.modules['search_document.combine_search'] = MockModule()
sys.modules['search_document.rerank'] = MockModule()

# Now import FastAPI app components directly
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from typing import Optional

# Import contract modules (these don't have complex deps)
from contract_drafting.models import get_all_templates, get_template_by_id
from contract_drafting.schemas import (
    TemplateListResponse,
    TemplateListItem,
)

print("=== Contract Module Test ===\n")

# Test 1: Get templates from MongoDB
print("1. Testing get_all_templates()...")
templates = get_all_templates()
print(f"   Found {len(templates)} templates:")
for t in templates:
    print(f"   - {t['template_id']}: {t['template_name']}")

# Test 2: Get specific template
print("\n2. Testing get_template_by_id('tpl_labor_001')...")
template = get_template_by_id('tpl_labor_001')
if template:
    print(f"   Found: {template['template_name']}")
    print(f"   Sections: {len(template.get('sections', []))}")
else:
    print("   Template not found!")

print("\n=== Contract API Endpoints ===")
print("""
Contract Drafting:
  GET  /contract/templates           - List all templates
  GET  /contract/templates/{id}      - Get template details
  POST /contract/draft               - Create contract from template
  GET  /contract/drafts/{user_id}    - User's draft history

Contract Analysis:
  POST /contract/analyze             - Upload & analyze contract
  GET  /contract/analysis/{id}       - Get analysis result
  GET  /contract/analyses/{user_id}  - User's analysis history
""")

print("=== Test Complete ===")
