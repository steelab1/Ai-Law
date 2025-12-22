"""
Contract Analysis Prompts
LLM prompts for contract analysis and comparison
"""

# System prompt for contract analyzer
CONTRACT_ANALYSIS_SYSTEM = """Ban la chuyen gia phap ly Viet Nam chuyen ve phan tich hop dong.
Nhiem vu cua ban la phan tich hop dong, tim cac diem bat loi, vi pham phap luat, va de xuat cai thien.

Nguyen tac:
1. Phan tich dua tren phap luat Viet Nam hien hanh
2. Chi ra cu the vi tri cac dieu khoan co van de
3. Danh gia khach quan, bao ve quyen loi cac ben
4. De xuat sua doi cu the, kha thi
5. Tham chieu dieu luat cu the khi co"""


# Detect contract type
CONTRACT_TYPE_DETECTION_PROMPT = """Phan tich noi dung hop dong sau va xac dinh loai hop dong:

---HOP DONG---
{contract_text}
---KET THUC---

Cac loai hop dong:
- labor: Hop dong lao dong
- sales: Hop dong mua ban
- rental: Hop dong thue (nha, xe, thiet bi...)
- services: Hop dong dich vu
- loan: Hop dong vay
- partnership: Hop dong hop tac kinh doanh
- other: Loai khac

Tra ve JSON:
{{
    "contract_type": "labor/sales/rental/services/loan/partnership/other",
    "confidence": 0.0-1.0,
    "contract_summary": "Tom tat ngan gon noi dung hop dong (2-3 cau)"
}}"""


# Detect unfavorable clauses
UNFAVORABLE_CLAUSE_PROMPT = """Phan tich hop dong sau va tim cac dieu khoan bat loi, gai bay, hoac co hai cho mot ben:

---HOP DONG---
{contract_text}
---KET THUC---

Loai hop dong: {contract_type}

Cac quy dinh phap luat lien quan:
{legal_context}

Tim cac dieu khoan:
1. Gai bay, an giau thong tin quan trong
2. Bat loi qua muc cho mot ben
3. Dieu khoan phat qua nang
4. Dieu khoan tu dong gia han hoac khong cho cham dut
5. Gioi han quyen khieu nai, khoi kien
6. Chuyen rui ro bat hop ly

Tra ve JSON:
{{
    "unfavorable_clauses": [
        {{
            "clause_text": "Noi dung dieu khoan",
            "location": "Dieu X, Khoan Y (neu xac dinh duoc)",
            "reason": "Ly do bat loi",
            "severity": "cao/trung_binh/thap",
            "affected_party": "Ben A/Ben B/Ca hai",
            "suggestion": "De xuat sua doi cu the",
            "legal_reference": "Dieu luat lien quan (neu co)"
        }}
    ]
}}"""


# Compare with standard template
STANDARD_COMPARISON_PROMPT = """So sanh hop dong voi mau chuan va quy dinh phap luat:

---HOP DONG CAN PHAN TICH---
{contract_text}
---KET THUC---

---MAU CHUAN---
{template_text}
---KET THUC---

Loai hop dong: {contract_type}

Cac quy dinh phap luat lien quan:
{legal_context}

Hay so sanh va tim:
1. Cac dieu khoan bat buoc bi thieu
2. Cac diem khac biet so voi mau chuan
3. Cac dieu khoan thua (khong can thiet)
4. Danh gia muc do phu hop (0-100%)

Tra ve JSON:
{{
    "compared_against": "Ten mau chuan",
    "missing_mandatory_clauses": [
        {{
            "clause_name": "Ten dieu khoan",
            "importance": "cao/trung_binh/thap",
            "suggested_text": "Noi dung de xuat",
            "legal_requirement": "Luat yeu cau (neu co)"
        }}
    ],
    "deviations_from_standard": ["Mo ta diem khac biet 1", "Mo ta 2"],
    "additional_non_standard_clauses": ["Dieu khoan thua 1"],
    "conformity_score": 75
}}"""


# Check legal compliance
COMPLIANCE_CHECK_PROMPT = """Kiem tra hop dong co tuan thu phap luat Viet Nam khong:

---HOP DONG---
{contract_text}
---KET THUC---

Loai hop dong: {contract_type}

Cac quy dinh phap luat lien quan:
{legal_context}

Kiem tra:
1. Hinh thuc hop dong (bang van ban, cong chung...)
2. Nang luc chu the (do tuoi, tham quyen ky...)
3. Noi dung khong vi pham dieu cam
4. Cac dieu khoan bat buoc theo luat
5. Thu tuc dang ky/thong bao (neu can)

Tra ve JSON:
{{
    "compliance_issues": [
        {{
            "issue": "Mo ta van de",
            "required_by": "Quy dinh yeu cau",
            "legal_reference": "Dieu X, Luat Y",
            "recommendation": "Khuyen nghi khac phuc",
            "severity": "cao/trung_binh/thap"
        }}
    ],
    "is_compliant": true/false
}}"""


# Summarize party obligations
OBLIGATIONS_SUMMARY_PROMPT = """Tong hop quyen va nghia vu cac ben trong hop dong:

---HOP DONG---
{contract_text}
---KET THUC---

Liet ke day du:
1. Tat ca nghia vu cua tung ben
2. Tat ca quyen loi cua tung ben
3. Danh gia can bang quyen/nghia vu

Tra ve JSON:
{{
    "party_obligations": [
        {{
            "party_name": "Ben A (Ten cu the neu co)",
            "obligations": ["Nghia vu 1", "Nghia vu 2"],
            "rights": ["Quyen 1", "Quyen 2"],
            "balance_assessment": "Nhan xet ve can bang quyen/nghia vu"
        }},
        {{
            "party_name": "Ben B",
            "obligations": [...],
            "rights": [...],
            "balance_assessment": "..."
        }}
    ]
}}"""


# Risk assessment
RISK_ASSESSMENT_PROMPT = """Danh gia rui ro tong the cua hop dong:

---HOP DONG---
{contract_text}
---KET THUC---

Loai hop dong: {contract_type}

Ket qua phan tich truoc do:
- Dieu khoan bat loi: {unfavorable_count}
- Van de tuan thu: {compliance_count}
- Diem phu hop mau: {conformity_score}%

Danh gia rui ro theo thang 0-10 (0=khong rui ro, 10=rui ro cuc ky cao):
1. Rui ro phap ly: Vi pham luat, tranh chap
2. Rui ro tai chinh: Thiet hai tien bac
3. Rui ro van hanh: Kho thuc hien
4. Rui ro danh tieng: Anh huong uy tin

Tra ve JSON:
{{
    "risk_assessment": {{
        "overall_score": 6.5,
        "risk_level": "cao/trung_binh/thap/toi_thieu",
        "breakdown": {{
            "legal_risk": 7.0,
            "financial_risk": 6.0,
            "operational_risk": 5.0,
            "reputation_risk": 4.0
        }},
        "key_risks": ["Rui ro 1", "Rui ro 2", "Rui ro 3"],
        "summary": "Tong ket danh gia rui ro 2-3 cau"
    }}
}}"""


# Improvement suggestions
IMPROVEMENT_SUGGESTIONS_PROMPT = """Dua tren phan tich hop dong, de xuat cai thien:

---HOP DONG---
{contract_text}
---KET THUC---

Loai hop dong: {contract_type}

Cac van de da phat hien:
{issues_summary}

De xuat cac cai thien cu the, kha thi:
1. Uu tien cac van de nghiem trong
2. De xuat text cu the (neu can)
3. Giai thich ly do cai thien

Tra ve JSON:
{{
    "improvement_suggestions": [
        {{
            "section": "Phan can cai thien",
            "current_state": "Tinh trang hien tai (neu can)",
            "suggestion": "De xuat cai thien cu the",
            "priority": "cao/trung_binh/thap",
            "rationale": "Ly do can cai thien"
        }}
    ]
}}"""


# RAG queries by contract type (for retrieving relevant laws)
RAG_QUERIES = {
    "labor": [
        "hop dong lao dong bat buoc",
        "quyen nguoi lao dong",
        "cham dut hop dong lao dong",
        "che do bao hiem xa hoi"
    ],
    "sales": [
        "hop dong mua ban bat buoc",
        "bao hanh san pham",
        "trach nhiem ben ban",
        "chuyen quyen so huu"
    ],
    "rental": [
        "hop dong thue nha bat buoc",
        "quyen nghia vu ben thue",
        "dat coc thue nha",
        "cham dut hop dong thue"
    ],
    "services": [
        "hop dong dich vu quy dinh",
        "chat luong dich vu",
        "boi thuong thiet hai dich vu"
    ],
    "loan": [
        "hop dong vay quy dinh",
        "lai suat toi da",
        "bao dam tien vay"
    ],
    "partnership": [
        "hop dong hop tac kinh doanh",
        "gop von kinh doanh",
        "phan chia loi nhuan"
    ]
}
