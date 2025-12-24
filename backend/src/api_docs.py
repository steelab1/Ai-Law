"""
API Documentation Strings
Tách riêng các docstrings và descriptions để app.py gọn hơn
"""

# =============================================================================
# APP METADATA
# =============================================================================

APP_TITLE = "Vietnamese Legal Q&A API"
APP_VERSION = "2.0.0"

APP_DESCRIPTION = """
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
"""

APP_TAGS = [
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


# =============================================================================
# LEGAL Q&A DOCS
# =============================================================================

RETRIEVAL_DOC = """
Tìm kiếm văn bản pháp luật liên quan đến câu hỏi.

**Mô tả:**
- Sử dụng kết hợp semantic search (paraphrase-vietnamese-law) và lexical search (Elasticsearch BM25)
- Áp dụng Reciprocal Rank Fusion (RRF) để xếp hạng kết quả
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

CHAT_COMPLETE_DOC = """
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

CHAT_COMPLETE_V2_DOC = """
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


# =============================================================================
# CONTRACT DRAFTING DOCS
# =============================================================================

LIST_TEMPLATES_DOC = """
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

TEMPLATE_DETAIL_DOC = """
Lấy chi tiết mẫu hợp đồng bao gồm các trường cần điền.

**Mô tả:**
- Trả về thông tin chi tiết của một mẫu hợp đồng
- Bao gồm các section, điều khoản, và danh sách placeholder cần điền

**Response bao gồm:**
- `sections`: Các phần của hợp đồng
- `legal_references`: Các điều luật tham chiếu
- `all_placeholders`: Danh sách các trường cần điền giá trị
"""

CREATE_DRAFT_DOC = """
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

USER_DRAFTS_DOC = """
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


# =============================================================================
# CONTRACT ANALYSIS DOCS
# =============================================================================

ANALYZE_CONTRACT_DOC = """
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

GET_ANALYSIS_DOC = """
Lấy kết quả phân tích đã lưu.

**Mô tả:**
- Truy xuất kết quả phân tích hợp đồng từ database
- Sử dụng `analysis_id` nhận được từ API `/contract/analyze`

**Lưu ý:**
- Trả về 404 nếu không tìm thấy kết quả
"""

USER_ANALYSES_DOC = """
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


# =============================================================================
# REQUEST EXAMPLES
# =============================================================================

COMPLETE_REQUEST_EXAMPLE = {
    "bot_id": "bot_Legal_VN",
    "user_id": "user_123456",
    "user_message": "Thời gian thử việc tối đa là bao lâu?",
    "sync_request": False
}

RETRIEVAL_REQUEST_EXAMPLE = {
    "query": "Thời gian thử việc tối đa là bao lâu?",
    "top_k_search": 30,
    "top_k_rerank": 5
}
