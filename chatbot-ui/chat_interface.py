import json
import logging
import os
import time
import requests

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

# Read backend URL from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")

# Page config
st.set_page_config(
    page_title="Legal AI - Tro ly Phap ly",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .contract-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background: white;
    }
    .risk-high { color: #dc3545; }
    .risk-medium { color: #ffc107; }
    .risk-low { color: #28a745; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal AI - Tro ly Phap ly Thong minh")

# Input bot
bot_id = "botLegal"
user_id = "1"


# ============================================
# API Functions
# ============================================

def send_user_request(text):
    url = f"{BACKEND_URL}/chat/complete"
    payload = json.dumps({
        "user_message": text,
        "user_id": str(user_id),
        "bot_id": bot_id,
    })
    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
    if response.status_code != 200:
        raise TimeoutError(f"Request to bot fail: {response.text}")
    return json.loads(response.text)


def get_bot_response(request_id):
    url = f"{BACKEND_URL}/chat/complete_v2/{request_id}"
    response = requests.request("GET", url, headers={}, data="", timeout=120)
    if response.status_code != 200:
        raise TimeoutError(f"Get bot response fail: {response.text}")
    return response.status_code, json.loads(response.text)


def get_chat_complete(text):
    user_request = send_user_request(text)
    request_id = user_request["task_id"]
    status_code, chat_response = get_bot_response(request_id)
    if status_code == 200:
        task_status = chat_response.get('task_status', '')
        task_result = chat_response.get('task_result')
        if task_status == 'FAILURE':
            return "Xin loi, co loi xay ra khi xu ly yeu cau cua ban. Vui long thu lai."
        if task_result and isinstance(task_result, dict) and 'content' in task_result:
            return task_result['content']
        elif task_result:
            return str(task_result)
        else:
            return "Khong nhan duoc phan hoi tu he thong."
    else:
        raise TimeoutError("Request fail, try again please")


def get_templates():
    """Get list of contract templates"""
    try:
        response = requests.get(f"{BACKEND_URL}/contract/templates", timeout=10)
        if response.status_code == 200:
            return response.json().get("templates", [])
    except Exception as e:
        st.error(f"Loi tai templates: {e}")
    return []


def get_template_detail(template_id):
    """Get template detail with placeholders"""
    try:
        response = requests.get(f"{BACKEND_URL}/contract/templates/{template_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Loi tai chi tiet template: {e}")
    return None


def generate_contract(template_id, values, additional_clauses=None):
    """Generate contract from template"""
    try:
        payload = {
            "user_id": user_id,
            "template_id": template_id,
            "values": values,
            "additional_clauses": additional_clauses,
            "include_legal_references": True
        }
        response = requests.post(
            f"{BACKEND_URL}/contract/draft",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Loi tao hop dong: {e}")
    return None


def analyze_contract(file_bytes, filename):
    """Analyze uploaded contract"""
    try:
        files = {"file": (filename, file_bytes)}
        data = {"user_id": user_id}
        # Use longer timeout - analysis can take 30-60 seconds
        response = requests.post(
            f"{BACKEND_URL}/contract/analyze",
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        st.error("Request timeout. Phan tich hop dong mat qua nhieu thoi gian.")
    except requests.exceptions.ConnectionError as e:
        st.error(f"Khong the ket noi den server. Vui long kiem tra backend dang chay.")
    except Exception as e:
        st.error(f"Loi phan tich hop dong: {e}")
    return None


# Streamed response
def response_generator(user_message):
    res = get_chat_complete(user_message)
    for line in res.split("\n\n"):
        for sen in line.split("\n"):
            yield sen + '\n\n'
            time.sleep(0.05)
        yield '\n'
    return res


# ============================================
# Tabs
# ============================================

tab1, tab2, tab3 = st.tabs(["💬 Chat Phap ly", "📝 Soan thao Hop dong", "🔍 Phan tich Hop dong"])


# ============================================
# Tab 1: Chat
# ============================================
with tab1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Nhap cau hoi phap ly cua ban..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})


# ============================================
# Tab 2: Contract Drafting
# ============================================
with tab2:
    st.header("Soan thao Hop dong")

    # Initialize session state for drafting
    if "drafting_step" not in st.session_state:
        st.session_state.drafting_step = 1
    if "selected_template" not in st.session_state:
        st.session_state.selected_template = None
    if "generated_contract" not in st.session_state:
        st.session_state.generated_contract = None

    # Step indicator
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.drafting_step >= 1:
            st.success("✅ Buoc 1: Chon mau")
        else:
            st.info("1️⃣ Buoc 1: Chon mau")
    with col2:
        if st.session_state.drafting_step >= 2:
            st.success("✅ Buoc 2: Nhap thong tin")
        elif st.session_state.drafting_step == 2:
            st.info("2️⃣ Buoc 2: Nhap thong tin")
        else:
            st.write("2️⃣ Buoc 2: Nhap thong tin")
    with col3:
        if st.session_state.drafting_step >= 3:
            st.success("✅ Buoc 3: Xem ket qua")
        else:
            st.write("3️⃣ Buoc 3: Xem ket qua")

    st.divider()

    # Step 1: Select template
    if st.session_state.drafting_step == 1:
        st.subheader("Chon mau hop dong")

        templates = get_templates()

        if templates:
            # Display templates in grid
            cols = st.columns(3)
            for idx, template in enumerate(templates):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background: #f9f9f9;">
                            <h4>{template.get('template_name', 'N/A')}</h4>
                            <p style="color: #666; font-size: 14px;">{template.get('description', '')}</p>
                            <p style="color: #888; font-size: 12px;">📋 {template.get('placeholder_count', 0)} truong can dien</p>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"Chon", key=f"select_{template['template_id']}"):
                            st.session_state.selected_template = get_template_detail(template['template_id'])
                            st.session_state.drafting_step = 2
                            st.rerun()
        else:
            st.warning("Chua co mau hop dong nao. Vui long chay seed_templates truoc.")
            st.code("cd backend/src && python -m contract_drafting.seed_templates")

    # Step 2: Fill form
    elif st.session_state.drafting_step == 2:
        template = st.session_state.selected_template

        if template:
            st.subheader(f"Nhap thong tin: {template.get('template_name', '')}")

            if st.button("← Quay lai chon mau"):
                st.session_state.drafting_step = 1
                st.session_state.selected_template = None
                st.rerun()

            st.divider()

            with st.form("contract_form"):
                values = {}
                placeholders = template.get("all_placeholders", [])

                # Create form fields in 2 columns
                col1, col2 = st.columns(2)

                for idx, ph in enumerate(placeholders):
                    with col1 if idx % 2 == 0 else col2:
                        key = ph.get("key", "")
                        label = ph.get("label", key)
                        ph_type = ph.get("type", "string")
                        required = ph.get("required", False)
                        default = ph.get("default_value", "")
                        hint = ph.get("hint", "")

                        label_text = f"{label} {'*' if required else ''}"

                        if ph_type == "date":
                            values[key] = st.date_input(label_text, help=hint)
                        elif ph_type == "number" or ph_type == "currency":
                            values[key] = st.number_input(label_text, value=0, help=hint)
                        else:
                            values[key] = st.text_input(label_text, value=default, help=hint)

                st.divider()
                additional = st.text_area(
                    "Yeu cau bo sung (tuy chon)",
                    placeholder="Nhap cac dieu khoan bo sung hoac yeu cau dac biet..."
                )

                submitted = st.form_submit_button("🚀 Tao Hop dong", type="primary")

                if submitted:
                    # Convert date objects to string
                    for k, v in values.items():
                        if hasattr(v, 'strftime'):
                            values[k] = v.strftime("%d/%m/%Y")
                        elif isinstance(v, (int, float)):
                            values[k] = str(v)

                    with st.spinner("AI dang soan thao hop dong..."):
                        result = generate_contract(
                            template['template_id'],
                            values,
                            additional if additional else None
                        )

                        if result:
                            st.session_state.generated_contract = result
                            st.session_state.drafting_step = 3
                            st.rerun()

    # Step 3: Preview result
    elif st.session_state.drafting_step == 3:
        st.subheader("Hop dong da tao")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📝 Tao hop dong moi"):
                st.session_state.drafting_step = 1
                st.session_state.selected_template = None
                st.session_state.generated_contract = None
                st.rerun()

        st.divider()

        contract = st.session_state.generated_contract
        if contract:
            content = contract.get("contract_content", "")

            # Display contract
            st.text_area("Noi dung hop dong", value=content, height=500)

            # Download button
            st.download_button(
                label="📥 Tai xuong (.txt)",
                data=content,
                file_name="hop-dong.txt",
                mime="text/plain"
            )

            # Show legal references if available
            refs = contract.get("legal_references_used", [])
            if refs:
                with st.expander("📚 Luat tham chieu"):
                    for ref in refs:
                        st.write(f"- {ref}")


# ============================================
# Tab 3: Contract Analysis
# ============================================
with tab3:
    st.header("Phan tich Hop dong")

    # Initialize session state
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    # File upload
    uploaded_file = st.file_uploader(
        "Upload hop dong (PDF, DOCX, DOC)",
        type=["pdf", "docx", "doc"],
        help="Toi da 10MB"
    )

    if uploaded_file:
        st.success(f"Da chon file: {uploaded_file.name}")

        if st.button("🔍 Phan tich hop dong", type="primary"):
            with st.spinner("AI dang phan tich hop dong... (15-30 giay)"):
                result = analyze_contract(uploaded_file.getvalue(), uploaded_file.name)
                if result:
                    st.session_state.analysis_result = result
                    st.rerun()

    # Display analysis result
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result.get("result", {})

        st.divider()

        # Summary
        st.subheader("📋 Tom tat")
        st.write(result.get("contract_summary", "Khong co tom tat"))

        # Risk Assessment
        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("📊 Danh gia rui ro")
            risk = result.get("risk_assessment", {})
            score = risk.get("overall_score", 0)

            # Risk score with color
            if score >= 7:
                st.error(f"⚠️ Diem rui ro: {score}/10")
            elif score >= 4:
                st.warning(f"⚡ Diem rui ro: {score}/10")
            else:
                st.success(f"✅ Diem rui ro: {score}/10")

            st.progress(score / 10)

            # Breakdown
            breakdown = risk.get("breakdown", {})
            if breakdown:
                st.write(f"- Rui ro phap ly: {breakdown.get('legal_risk', 'N/A')}")
                st.write(f"- Rui ro tai chinh: {breakdown.get('financial_risk', 'N/A')}")
                st.write(f"- Rui ro van hanh: {breakdown.get('operational_risk', 'N/A')}")

            # Conformity
            std = result.get("standard_comparison") or {}
            conformity = std.get("conformity_score", 0) if std else 0
            st.metric("Phu hop mau chuan", f"{conformity}%")

        with col1:
            # Unfavorable clauses
            st.subheader("⚠️ Dieu khoan bat loi")
            unfavorable = result.get("unfavorable_clauses", [])
            if unfavorable:
                for clause in unfavorable:
                    with st.expander(f"🔴 {clause.get('location', 'Dieu khoan')} - {clause.get('severity', '')}"):
                        st.write(f"**Van de:** {clause.get('reason', '')}")
                        st.write(f"**De xuat:** {clause.get('suggestion', '')}")
            else:
                st.success("Khong tim thay dieu khoan bat loi")

            # Compliance issues
            st.subheader("⚖️ Van de tuan thu phap luat")
            compliance = result.get("compliance_issues", [])
            if compliance:
                for issue in compliance:
                    with st.expander(f"🟡 {issue.get('issue', '')}"):
                        st.write(f"**Yeu cau boi:** {issue.get('required_by', '')}")
                        st.write(f"**Tham chieu:** {issue.get('legal_reference', '')}")
            else:
                st.success("Khong co van de tuan thu")

            # Improvement suggestions
            st.subheader("💡 De xuat cai thien")
            suggestions = result.get("improvement_suggestions", [])
            if suggestions:
                for sug in suggestions:
                    with st.expander(f"🟢 {sug.get('section', '')}"):
                        st.write(sug.get('suggestion', ''))
            else:
                st.info("Khong co de xuat")

        # New analysis button
        if st.button("🔄 Phan tich hop dong khac"):
            st.session_state.analysis_result = None
            st.rerun()
