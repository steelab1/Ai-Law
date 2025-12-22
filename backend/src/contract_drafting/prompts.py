"""
Contract Drafting Prompts
LLM prompts for contract generation and enhancement
"""

# System prompt for contract drafting assistant
CONTRACT_DRAFTING_SYSTEM = """Ban la chuyen gia phap ly Viet Nam chuyen ve soan thao hop dong.
Nhiem vu cua ban la tao ra hop dong day du, chinh xac ve mat phap ly dua tren mau co san.

Nguyen tac:
1. Dam bao tuan thu phap luat Viet Nam hien hanh
2. Su dung ngon ngu phap ly chinh xac, ro rang
3. Bao ve quyen loi cong bang cho cac ben
4. Tranh cac dieu khoan mo ho hoac bat loi
5. Tham chieu cac dieu luat lien quan khi can thiet"""


# Prompt to enhance contract with legal context
ENHANCE_CONTRACT_PROMPT = """Dua tren hop dong mau da duoc dien thong tin:

---HOP DONG MAU---
{contract_draft}
---KET THUC---

Va cac quy dinh phap luat lien quan:

---LUAT LIEN QUAN---
{legal_context}
---KET THUC---

Hay:
1. Kiem tra tinh hop phap cua cac dieu khoan
2. Bo sung cac dieu khoan bat buoc con thieu (neu co)
3. Dieu chinh ngon ngu cho phu hop voi thong le phap ly
4. Them tham chieu luat cu the o nhung cho can thiet

{additional_requirements}

Tra ve hop dong hoan chinh. Chi tra ve noi dung hop dong, khong giai thich them."""


# Prompt to generate additional clauses
GENERATE_CLAUSES_PROMPT = """Dua tren loai hop dong: {contract_type}
Va yeu cau bo sung: {requirements}

Cac quy dinh phap luat lien quan:
{legal_context}

Hay tao cac dieu khoan bo sung phu hop. Moi dieu khoan can:
1. Co ten dieu khoan
2. Noi dung chi tiet
3. Tham chieu luat (neu co)

Tra ve dang JSON:
{{
    "additional_clauses": [
        {{
            "clause_name": "Ten dieu khoan",
            "content": "Noi dung dieu khoan",
            "legal_reference": "Dieu X, Luat Y (neu co)"
        }}
    ]
}}"""


# Prompt to validate contract compliance
VALIDATE_CONTRACT_PROMPT = """Kiem tra hop dong sau co tuan thu phap luat Viet Nam khong:

---HOP DONG---
{contract_content}
---KET THUC---

Loai hop dong: {contract_type}

Cac quy dinh phap luat lien quan:
{legal_context}

Tra ve dang JSON:
{{
    "is_compliant": true/false,
    "issues": [
        {{
            "clause": "Dieu khoan co van de",
            "issue": "Mo ta van de",
            "legal_reference": "Dieu luat vi pham",
            "suggestion": "De xuat sua doi"
        }}
    ],
    "warnings": ["Canh bao 1", "Canh bao 2"],
    "missing_mandatory_clauses": ["Dieu khoan bat buoc con thieu"]
}}"""


# RAG queries by contract type
RAG_QUERIES = {
    "labor": [
        "quy dinh hop dong lao dong",
        "nghia vu nguoi su dung lao dong",
        "quyen loi nguoi lao dong",
        "bao hiem xa hoi bat buoc",
        "cham dut hop dong lao dong"
    ],
    "sales": [
        "hop dong mua ban hang hoa",
        "nghia vu ben ban ben mua",
        "bao hanh san pham",
        "chuyen quyen so huu",
        "phat vi pham hop dong"
    ],
    "rental": [
        "hop dong thue nha o",
        "quyen va nghia vu ben thue",
        "dat coc thue nha",
        "cham dut hop dong thue",
        "sua chua tai san thue"
    ],
    "services": [
        "hop dong dich vu",
        "nghia vu ben cung cap dich vu",
        "chat luong dich vu",
        "thanh toan dich vu",
        "boi thuong thiet hai"
    ],
    "loan": [
        "hop dong vay tien",
        "lai suat cho vay",
        "bao dam tien vay",
        "tra no truoc han",
        "xu ly no qua han"
    ],
    "partnership": [
        "hop dong hop tac kinh doanh",
        "gop von kinh doanh",
        "phan chia loi nhuan",
        "quyen va nghia vu cac ben",
        "cham dut hop tac"
    ]
}


# Legal references by contract type
LEGAL_REFERENCES = {
    "labor": [
        "Bo luat Lao dong 2019",
        "Luat Bao hiem xa hoi 2014",
        "Nghi dinh 145/2020/ND-CP"
    ],
    "sales": [
        "Bo luat Dan su 2015",
        "Luat Thuong mai 2005",
        "Luat Bao ve quyen loi nguoi tieu dung"
    ],
    "rental": [
        "Bo luat Dan su 2015",
        "Luat Nha o 2014",
        "Luat Kinh doanh bat dong san 2014"
    ],
    "services": [
        "Bo luat Dan su 2015",
        "Luat Thuong mai 2005"
    ],
    "loan": [
        "Bo luat Dan su 2015",
        "Luat Cac to chuc tin dung 2010"
    ],
    "partnership": [
        "Bo luat Dan su 2015",
        "Luat Doanh nghiep 2020",
        "Luat Dau tu 2020"
    ]
}
