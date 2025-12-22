"""
Seed Initial Contract Templates
Run this script to populate MongoDB with contract templates
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contract_drafting.models import save_template


# =============================================================================
# LABOR CONTRACT TEMPLATE
# =============================================================================

LABOR_CONTRACT_TEMPLATE = {
    "template_id": "tpl_labor_001",
    "template_type": "labor",
    "template_name": "Hop dong lao dong co thoi han",
    "description": "Mau hop dong lao dong co thoi han theo quy dinh Bo luat Lao dong 2019",
    "industry": "general",
    "sections": [
        {
            "section_id": "header",
            "section_name": "Phan mo dau",
            "order": 1,
            "content_template": """CONG HOA XA HOI CHU NGHIA VIET NAM
Doc lap - Tu do - Hanh phuc
-------------------

HOP DONG LAO DONG
So: {{contract_number}}

Can cu Bo luat Lao dong so 45/2019/QH14 ngay 20/11/2019;
Can cu nhu cau va kha nang cua cac ben;

Hom nay, ngay {{contract_date}}, tai {{contract_location}}, chung toi gom:""",
            "placeholders": [
                {"key": "contract_number", "label": "So hop dong", "type": "string", "required": True},
                {"key": "contract_date", "label": "Ngay ky hop dong", "type": "date", "required": True},
                {"key": "contract_location", "label": "Dia diem ky", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_a",
            "section_name": "Thong tin Ben A (Nguoi su dung lao dong)",
            "order": 2,
            "content_template": """BEN A (NGUOI SU DUNG LAO DONG):
Ten don vi: {{party_a_name}}
Dia chi: {{party_a_address}}
Ma so thue: {{party_a_tax_code}}
Dien thoai: {{party_a_phone}}
Dai dien boi: {{party_a_representative}}
Chuc vu: {{party_a_position}}""",
            "placeholders": [
                {"key": "party_a_name", "label": "Ten cong ty", "type": "string", "required": True},
                {"key": "party_a_address", "label": "Dia chi cong ty", "type": "string", "required": True},
                {"key": "party_a_tax_code", "label": "Ma so thue", "type": "string", "required": True},
                {"key": "party_a_phone", "label": "Dien thoai", "type": "string", "required": False},
                {"key": "party_a_representative", "label": "Nguoi dai dien", "type": "string", "required": True},
                {"key": "party_a_position", "label": "Chuc vu nguoi dai dien", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_b",
            "section_name": "Thong tin Ben B (Nguoi lao dong)",
            "order": 3,
            "content_template": """BEN B (NGUOI LAO DONG):
Ho va ten: {{party_b_name}}
Ngay sinh: {{party_b_dob}}
Gioi tinh: {{party_b_gender}}
So CCCD/CMND: {{party_b_id_number}}
Ngay cap: {{party_b_id_date}}    Noi cap: {{party_b_id_place}}
Dia chi thuong tru: {{party_b_address}}
Trinh do chuyen mon: {{party_b_qualification}}""",
            "placeholders": [
                {"key": "party_b_name", "label": "Ho ten nguoi lao dong", "type": "string", "required": True},
                {"key": "party_b_dob", "label": "Ngay sinh", "type": "date", "required": True},
                {"key": "party_b_gender", "label": "Gioi tinh", "type": "string", "required": True},
                {"key": "party_b_id_number", "label": "So CCCD/CMND", "type": "string", "required": True},
                {"key": "party_b_id_date", "label": "Ngay cap CCCD", "type": "date", "required": True},
                {"key": "party_b_id_place", "label": "Noi cap CCCD", "type": "string", "required": True},
                {"key": "party_b_address", "label": "Dia chi thuong tru", "type": "string", "required": True},
                {"key": "party_b_qualification", "label": "Trinh do chuyen mon", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_1",
            "section_name": "Dieu 1: Cong viec va dia diem lam viec",
            "order": 4,
            "content_template": """DIEU 1: CONG VIEC VA DIA DIEM LAM VIEC

1.1. Loai hop dong lao dong: {{contract_type_detail}}
1.2. Thoi han hop dong: Tu ngay {{start_date}} den ngay {{end_date}}
1.3. Cong viec phai lam: {{job_description}}
1.4. Chuc danh chuyen mon: {{job_title}}
1.5. Dia diem lam viec: {{work_location}}""",
            "placeholders": [
                {"key": "contract_type_detail", "label": "Loai hop dong (xac dinh/khong xac dinh thoi han)", "type": "string", "required": True, "default_value": "Xac dinh thoi han"},
                {"key": "start_date", "label": "Ngay bat dau", "type": "date", "required": True},
                {"key": "end_date", "label": "Ngay ket thuc", "type": "date", "required": True},
                {"key": "job_description", "label": "Mo ta cong viec", "type": "string", "required": True},
                {"key": "job_title", "label": "Chuc danh", "type": "string", "required": True},
                {"key": "work_location", "label": "Dia diem lam viec", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_2",
            "section_name": "Dieu 2: Thoi gio lam viec va nghi ngoi",
            "order": 5,
            "content_template": """DIEU 2: THOI GIO LAM VIEC VA NGHI NGOI

2.1. Thoi gio lam viec: {{working_hours}} gio/ngay, {{working_days}} ngay/tuan
2.2. Thoi gian nghi ngoi: Theo quy dinh cua phap luat va noi quy cong ty
2.3. Nghi phep nam: {{annual_leave}} ngay/nam (huong nguyen luong)
2.4. Nghi le, Tet: Theo quy dinh cua phap luat""",
            "placeholders": [
                {"key": "working_hours", "label": "So gio lam viec/ngay", "type": "number", "required": True, "default_value": "8"},
                {"key": "working_days", "label": "So ngay lam viec/tuan", "type": "number", "required": True, "default_value": "5"},
                {"key": "annual_leave", "label": "So ngay phep nam", "type": "number", "required": True, "default_value": "12"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_3",
            "section_name": "Dieu 3: Luong, phu cap va cac khoan bo sung",
            "order": 6,
            "content_template": """DIEU 3: LUONG, PHU CAP VA CAC KHOAN BO SUNG

3.1. Muc luong chinh: {{base_salary}} dong/thang
3.2. Phu cap (neu co):
    - Phu cap an trua: {{lunch_allowance}} dong/thang
    - Phu cap xang xe: {{transport_allowance}} dong/thang
    - Phu cap dien thoai: {{phone_allowance}} dong/thang
3.3. Hinh thuc tra luong: {{payment_method}}
3.4. Ky han tra luong: {{payment_period}}
3.5. Che do nang luong: Theo quy che luong cua cong ty""",
            "placeholders": [
                {"key": "base_salary", "label": "Luong co ban (VND)", "type": "currency", "required": True},
                {"key": "lunch_allowance", "label": "Phu cap an trua", "type": "currency", "required": False, "default_value": "0"},
                {"key": "transport_allowance", "label": "Phu cap xang xe", "type": "currency", "required": False, "default_value": "0"},
                {"key": "phone_allowance", "label": "Phu cap dien thoai", "type": "currency", "required": False, "default_value": "0"},
                {"key": "payment_method", "label": "Hinh thuc tra luong", "type": "string", "required": True, "default_value": "Chuyen khoan ngan hang"},
                {"key": "payment_period", "label": "Ky han tra luong", "type": "string", "required": True, "default_value": "Ngay 05 hang thang"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_4",
            "section_name": "Dieu 4: Bao hiem xa hoi, bao hiem y te, bao hiem that nghiep",
            "order": 7,
            "content_template": """DIEU 4: BAO HIEM XA HOI, BAO HIEM Y TE, BAO HIEM THAT NGHIEP

4.1. Ben A co trach nhiem dong bao hiem xa hoi, bao hiem y te, bao hiem that nghiep cho Ben B theo quy dinh cua phap luat.
4.2. Ty le dong:
    - Bao hiem xa hoi: Ben A {{bhxh_employer}}%, Ben B {{bhxh_employee}}%
    - Bao hiem y te: Ben A {{bhyt_employer}}%, Ben B {{bhyt_employee}}%
    - Bao hiem that nghiep: Ben A {{bhtn_employer}}%, Ben B {{bhtn_employee}}%
4.3. Muc luong dong bao hiem: Theo quy dinh phap luat hien hanh""",
            "placeholders": [
                {"key": "bhxh_employer", "label": "BHXH - ty le cong ty", "type": "number", "required": True, "default_value": "17.5"},
                {"key": "bhxh_employee", "label": "BHXH - ty le nhan vien", "type": "number", "required": True, "default_value": "8"},
                {"key": "bhyt_employer", "label": "BHYT - ty le cong ty", "type": "number", "required": True, "default_value": "3"},
                {"key": "bhyt_employee", "label": "BHYT - ty le nhan vien", "type": "number", "required": True, "default_value": "1.5"},
                {"key": "bhtn_employer", "label": "BHTN - ty le cong ty", "type": "number", "required": True, "default_value": "1"},
                {"key": "bhtn_employee", "label": "BHTN - ty le nhan vien", "type": "number", "required": True, "default_value": "1"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_5",
            "section_name": "Dieu 5: Quyen va nghia vu cua cac ben",
            "order": 8,
            "content_template": """DIEU 5: QUYEN VA NGHIA VU CUA CAC BEN

5.1. Quyen va nghia vu cua Ben A:
a) Quyen:
- Dieu hanh, quan ly nguoi lao dong theo phap luat va noi quy cong ty
- Xu ly vi pham ky luat lao dong theo quy dinh
- Cham dut hop dong theo quy dinh phap luat

b) Nghia vu:
- Bao dam viec lam va thuc hien day du cac dieu khoan hop dong
- Tra luong, phu cap dung thoi han
- Dong bao hiem xa hoi, bao hiem y te, bao hiem that nghiep
- Dam bao dieu kien lam viec an toan, ve sinh lao dong

5.2. Quyen va nghia vu cua Ben B:
a) Quyen:
- Duoc tra luong day du, dung han
- Duoc tham gia bao hiem xa hoi, y te, that nghiep
- Nghi phep, nghi le theo quy dinh
- Don phuong cham dut hop dong theo quy dinh phap luat

b) Nghia vu:
- Hoan thanh cong viec theo hop dong
- Chap hanh noi quy, quy che cong ty
- Bao ve tai san, bi mat kinh doanh cua cong ty""",
            "placeholders": [],
            "is_optional": False
        },
        {
            "section_id": "article_6",
            "section_name": "Dieu 6: Cham dut hop dong lao dong",
            "order": 9,
            "content_template": """DIEU 6: CHAM DUT HOP DONG LAO DONG

6.1. Hop dong lao dong cham dut trong cac truong hop:
- Het han hop dong
- Hoan thanh cong viec theo hop dong
- Hai ben thoa thuan cham dut
- Nguoi lao dong bi xu ly ky luat sa thai
- Don phuong cham dut hop dong theo quy dinh phap luat
- Cac truong hop khac theo quy dinh Bo luat Lao dong

6.2. Thoi han bao truoc khi don phuong cham dut hop dong: {{notice_period}} ngay

6.3. Che do khi cham dut hop dong: Theo quy dinh phap luat hien hanh""",
            "placeholders": [
                {"key": "notice_period", "label": "Thoi han bao truoc (ngay)", "type": "number", "required": True, "default_value": "30"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_7",
            "section_name": "Dieu 7: Dieu khoan chung",
            "order": 10,
            "content_template": """DIEU 7: DIEU KHOAN CHUNG

7.1. Hop dong nay co hieu luc ke tu ngay ky.
7.2. Hop dong nay duoc lap thanh 02 ban co gia tri phap ly nhu nhau, moi ben giu 01 ban.
7.3. Khi phat sinh tranh chap, hai ben giai quyet bang thuong luong. Neu khong dat duoc thoa thuan, tranh chap se duoc giai quyet tai co quan co tham quyen.
7.4. Cac van de khac khong quy dinh trong hop dong nay se duoc thuc hien theo Bo luat Lao dong va cac van ban phap luat hien hanh.

{{additional_terms}}""",
            "placeholders": [
                {"key": "additional_terms", "label": "Dieu khoan bo sung (neu co)", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "signature",
            "section_name": "Chu ky",
            "order": 11,
            "content_template": """
DAI DIEN BEN A                                    BEN B
(Ky, ghi ro ho ten, dong dau)                    (Ky, ghi ro ho ten)




{{party_a_representative}}                        {{party_b_name}}""",
            "placeholders": [],
            "is_optional": False
        }
    ],
    "rag_queries": [
        "quy dinh hop dong lao dong",
        "quyen loi nguoi lao dong",
        "nghia vu nguoi su dung lao dong",
        "bao hiem xa hoi bat buoc",
        "cham dut hop dong lao dong"
    ],
    "legal_references": [
        "Bo luat Lao dong 2019",
        "Luat Bao hiem xa hoi 2014",
        "Nghi dinh 145/2020/ND-CP"
    ]
}


# =============================================================================
# SALES CONTRACT TEMPLATE
# =============================================================================

SALES_CONTRACT_TEMPLATE = {
    "template_id": "tpl_sales_001",
    "template_type": "sales",
    "template_name": "Hop dong mua ban hang hoa",
    "description": "Mau hop dong mua ban hang hoa theo quy dinh Bo luat Dan su 2015 va Luat Thuong mai 2005",
    "industry": "general",
    "sections": [
        {
            "section_id": "header",
            "section_name": "Phan mo dau",
            "order": 1,
            "content_template": """CONG HOA XA HOI CHU NGHIA VIET NAM
Doc lap - Tu do - Hanh phuc
-------------------

HOP DONG MUA BAN HANG HOA
So: {{contract_number}}

Can cu Bo luat Dan su so 91/2015/QH13 ngay 24/11/2015;
Can cu Luat Thuong mai so 36/2005/QH11 ngay 14/6/2005;
Can cu nhu cau va kha nang cua cac ben;

Hom nay, ngay {{contract_date}}, tai {{contract_location}}, chung toi gom:""",
            "placeholders": [
                {"key": "contract_number", "label": "So hop dong", "type": "string", "required": True},
                {"key": "contract_date", "label": "Ngay ky hop dong", "type": "date", "required": True},
                {"key": "contract_location", "label": "Dia diem ky", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_a",
            "section_name": "Thong tin Ben Ban",
            "order": 2,
            "content_template": """BEN BAN (BEN A):
Ten doanh nghiep/ca nhan: {{party_a_name}}
Dia chi: {{party_a_address}}
Ma so thue: {{party_a_tax_code}}
So tai khoan: {{party_a_bank_account}}
Ngan hang: {{party_a_bank_name}}
Dai dien boi: {{party_a_representative}}
Chuc vu: {{party_a_position}}
Dien thoai: {{party_a_phone}}""",
            "placeholders": [
                {"key": "party_a_name", "label": "Ten ben ban", "type": "string", "required": True},
                {"key": "party_a_address", "label": "Dia chi", "type": "string", "required": True},
                {"key": "party_a_tax_code", "label": "Ma so thue", "type": "string", "required": True},
                {"key": "party_a_bank_account", "label": "So tai khoan", "type": "string", "required": True},
                {"key": "party_a_bank_name", "label": "Ngan hang", "type": "string", "required": True},
                {"key": "party_a_representative", "label": "Nguoi dai dien", "type": "string", "required": True},
                {"key": "party_a_position", "label": "Chuc vu", "type": "string", "required": True},
                {"key": "party_a_phone", "label": "Dien thoai", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_b",
            "section_name": "Thong tin Ben Mua",
            "order": 3,
            "content_template": """BEN MUA (BEN B):
Ten doanh nghiep/ca nhan: {{party_b_name}}
Dia chi: {{party_b_address}}
Ma so thue: {{party_b_tax_code}}
So tai khoan: {{party_b_bank_account}}
Ngan hang: {{party_b_bank_name}}
Dai dien boi: {{party_b_representative}}
Chuc vu: {{party_b_position}}
Dien thoai: {{party_b_phone}}""",
            "placeholders": [
                {"key": "party_b_name", "label": "Ten ben mua", "type": "string", "required": True},
                {"key": "party_b_address", "label": "Dia chi", "type": "string", "required": True},
                {"key": "party_b_tax_code", "label": "Ma so thue", "type": "string", "required": True},
                {"key": "party_b_bank_account", "label": "So tai khoan", "type": "string", "required": True},
                {"key": "party_b_bank_name", "label": "Ngan hang", "type": "string", "required": True},
                {"key": "party_b_representative", "label": "Nguoi dai dien", "type": "string", "required": True},
                {"key": "party_b_position", "label": "Chuc vu", "type": "string", "required": True},
                {"key": "party_b_phone", "label": "Dien thoai", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_1",
            "section_name": "Dieu 1: Hang hoa mua ban",
            "order": 4,
            "content_template": """DIEU 1: HANG HOA MUA BAN

1.1. Ten hang hoa: {{product_name}}
1.2. Quy cach, chung loai: {{product_specs}}
1.3. So luong: {{quantity}} {{unit}}
1.4. Chat luong: {{quality_standard}}
1.5. Xuat xu: {{origin}}
1.6. Bao bi, dong goi: {{packaging}}""",
            "placeholders": [
                {"key": "product_name", "label": "Ten hang hoa", "type": "string", "required": True},
                {"key": "product_specs", "label": "Quy cach, chung loai", "type": "string", "required": True},
                {"key": "quantity", "label": "So luong", "type": "number", "required": True},
                {"key": "unit", "label": "Don vi tinh", "type": "string", "required": True},
                {"key": "quality_standard", "label": "Tieu chuan chat luong", "type": "string", "required": True},
                {"key": "origin", "label": "Xuat xu", "type": "string", "required": True, "default_value": "Viet Nam"},
                {"key": "packaging", "label": "Quy cach dong goi", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_2",
            "section_name": "Dieu 2: Gia ca va phuong thuc thanh toan",
            "order": 5,
            "content_template": """DIEU 2: GIA CA VA PHUONG THUC THANH TOAN

2.1. Don gia: {{unit_price}} dong/{{unit}}
2.2. Tong gia tri hop dong: {{total_value}} dong
     (Bang chu: {{total_value_text}})
2.3. Gia tren da bao gom/chua bao gom VAT: {{vat_included}}
2.4. Phuong thuc thanh toan: {{payment_method}}
2.5. Thoi han thanh toan: {{payment_terms}}
2.6. Dong tien thanh toan: {{currency}}""",
            "placeholders": [
                {"key": "unit_price", "label": "Don gia", "type": "currency", "required": True},
                {"key": "total_value", "label": "Tong gia tri", "type": "currency", "required": True},
                {"key": "total_value_text", "label": "Tong gia tri bang chu", "type": "string", "required": True},
                {"key": "vat_included", "label": "Bao gom VAT?", "type": "string", "required": True, "default_value": "Chua bao gom VAT 10%"},
                {"key": "payment_method", "label": "Phuong thuc thanh toan", "type": "string", "required": True, "default_value": "Chuyen khoan ngan hang"},
                {"key": "payment_terms", "label": "Thoi han thanh toan", "type": "string", "required": True},
                {"key": "currency", "label": "Dong tien", "type": "string", "required": True, "default_value": "VND (Dong Viet Nam)"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_3",
            "section_name": "Dieu 3: Giao nhan hang",
            "order": 6,
            "content_template": """DIEU 3: GIAO NHAN HANG

3.1. Thoi gian giao hang: {{delivery_date}}
3.2. Dia diem giao hang: {{delivery_location}}
3.3. Phuong thuc giao hang: {{delivery_method}}
3.4. Chi phi van chuyen: {{shipping_cost}}
3.5. Thu tuc giao nhan:
    - Ben A giao hang kem theo: Hoa don, phieu xuat kho, giay chung nhan chat luong (neu co)
    - Ben B kiem tra hang hoa va ky nhan
3.6. Chuyen giao rui ro: Tu thoi diem giao hang cho Ben B tai dia diem giao hang""",
            "placeholders": [
                {"key": "delivery_date", "label": "Thoi gian giao hang", "type": "string", "required": True},
                {"key": "delivery_location", "label": "Dia diem giao hang", "type": "string", "required": True},
                {"key": "delivery_method", "label": "Phuong thuc giao hang", "type": "string", "required": True},
                {"key": "shipping_cost", "label": "Chi phi van chuyen", "type": "string", "required": True, "default_value": "Ben Ban chiu"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_4",
            "section_name": "Dieu 4: Bao hanh",
            "order": 7,
            "content_template": """DIEU 4: BAO HANH

4.1. Thoi gian bao hanh: {{warranty_period}}
4.2. Dieu kien bao hanh: Hang hoa bi loi do nha san xuat
4.3. Khong bao hanh trong truong hop:
    - Hang hoa bi hu hong do nguoi mua su dung sai huong dan
    - Hang hoa bi hu hong do thien tai, hoa hoan
    - Hang hoa da qua thoi han bao hanh
4.4. Phuong thuc bao hanh: {{warranty_method}}""",
            "placeholders": [
                {"key": "warranty_period", "label": "Thoi gian bao hanh", "type": "string", "required": True, "default_value": "12 thang"},
                {"key": "warranty_method", "label": "Phuong thuc bao hanh", "type": "string", "required": True, "default_value": "Sua chua hoac thay the mien phi"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_5",
            "section_name": "Dieu 5: Trach nhiem vi pham",
            "order": 8,
            "content_template": """DIEU 5: TRACH NHIEM VI PHAM

5.1. Neu Ben A cham giao hang: Phat {{late_delivery_penalty}}% gia tri lo hang cham giao cho moi ngay cham.
5.2. Neu Ben B cham thanh toan: Phat {{late_payment_penalty}}% gia tri cham tra cho moi ngay cham.
5.3. Neu hang hoa khong dat chat luong: Ben A phai doi tra hoac boi thuong thiet hai.
5.4. Muc phat toi da: {{max_penalty}}% gia tri hop dong.
5.5. Truong hop bat kha khang: Cac ben khong phai chiu trach nhiem boi thuong khi xay ra su kien bat kha khang theo quy dinh phap luat.""",
            "placeholders": [
                {"key": "late_delivery_penalty", "label": "Phat cham giao hang (%/ngay)", "type": "number", "required": True, "default_value": "0.5"},
                {"key": "late_payment_penalty", "label": "Phat cham thanh toan (%/ngay)", "type": "number", "required": True, "default_value": "0.5"},
                {"key": "max_penalty", "label": "Muc phat toi da (%)", "type": "number", "required": True, "default_value": "8"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_6",
            "section_name": "Dieu 6: Dieu khoan chung",
            "order": 9,
            "content_template": """DIEU 6: DIEU KHOAN CHUNG

6.1. Hop dong nay co hieu luc ke tu ngay ky.
6.2. Moi sua doi, bo sung phai duoc lap thanh van ban va co chu ky cua ca hai ben.
6.3. Khi phat sinh tranh chap, hai ben giai quyet bang thuong luong. Neu khong thanh, tranh chap se duoc giai quyet tai Toa an nhan dan co tham quyen.
6.4. Hop dong nay duoc lap thanh 02 ban co gia tri phap ly nhu nhau, moi ben giu 01 ban.

{{additional_terms}}""",
            "placeholders": [
                {"key": "additional_terms", "label": "Dieu khoan bo sung", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "signature",
            "section_name": "Chu ky",
            "order": 10,
            "content_template": """
DAI DIEN BEN BAN                                  DAI DIEN BEN MUA
(Ky, ghi ro ho ten, dong dau)                    (Ky, ghi ro ho ten, dong dau)




{{party_a_representative}}                        {{party_b_representative}}""",
            "placeholders": [],
            "is_optional": False
        }
    ],
    "rag_queries": [
        "hop dong mua ban hang hoa",
        "trach nhiem ben ban",
        "bao hanh san pham",
        "chuyen quyen so huu"
    ],
    "legal_references": [
        "Bo luat Dan su 2015",
        "Luat Thuong mai 2005"
    ]
}


# =============================================================================
# RENTAL CONTRACT TEMPLATE
# =============================================================================

RENTAL_CONTRACT_TEMPLATE = {
    "template_id": "tpl_rental_001",
    "template_type": "rental",
    "template_name": "Hop dong thue nha o",
    "description": "Mau hop dong thue nha o theo quy dinh Luat Nha o 2014 va Bo luat Dan su 2015",
    "industry": "real_estate",
    "sections": [
        {
            "section_id": "header",
            "section_name": "Phan mo dau",
            "order": 1,
            "content_template": """CONG HOA XA HOI CHU NGHIA VIET NAM
Doc lap - Tu do - Hanh phuc
-------------------

HOP DONG THUE NHA O
So: {{contract_number}}

Can cu Bo luat Dan su so 91/2015/QH13 ngay 24/11/2015;
Can cu Luat Nha o so 65/2014/QH13 ngay 25/11/2014;
Can cu nhu cau va kha nang cua cac ben;

Hom nay, ngay {{contract_date}}, tai {{contract_location}}, chung toi gom:""",
            "placeholders": [
                {"key": "contract_number", "label": "So hop dong", "type": "string", "required": True},
                {"key": "contract_date", "label": "Ngay ky", "type": "date", "required": True},
                {"key": "contract_location", "label": "Dia diem ky", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_a",
            "section_name": "Thong tin Ben Cho thue",
            "order": 2,
            "content_template": """BEN CHO THUE (BEN A):
Ho va ten: {{party_a_name}}
So CCCD/CMND: {{party_a_id}}
Ngay cap: {{party_a_id_date}}    Noi cap: {{party_a_id_place}}
Dia chi thuong tru: {{party_a_address}}
Dien thoai: {{party_a_phone}}""",
            "placeholders": [
                {"key": "party_a_name", "label": "Ho ten chu nha", "type": "string", "required": True},
                {"key": "party_a_id", "label": "So CCCD/CMND", "type": "string", "required": True},
                {"key": "party_a_id_date", "label": "Ngay cap", "type": "date", "required": True},
                {"key": "party_a_id_place", "label": "Noi cap", "type": "string", "required": True},
                {"key": "party_a_address", "label": "Dia chi", "type": "string", "required": True},
                {"key": "party_a_phone", "label": "Dien thoai", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "party_b",
            "section_name": "Thong tin Ben Thue",
            "order": 3,
            "content_template": """BEN THUE (BEN B):
Ho va ten: {{party_b_name}}
So CCCD/CMND: {{party_b_id}}
Ngay cap: {{party_b_id_date}}    Noi cap: {{party_b_id_place}}
Dia chi thuong tru: {{party_b_address}}
Dien thoai: {{party_b_phone}}""",
            "placeholders": [
                {"key": "party_b_name", "label": "Ho ten nguoi thue", "type": "string", "required": True},
                {"key": "party_b_id", "label": "So CCCD/CMND", "type": "string", "required": True},
                {"key": "party_b_id_date", "label": "Ngay cap", "type": "date", "required": True},
                {"key": "party_b_id_place", "label": "Noi cap", "type": "string", "required": True},
                {"key": "party_b_address", "label": "Dia chi", "type": "string", "required": True},
                {"key": "party_b_phone", "label": "Dien thoai", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_1",
            "section_name": "Dieu 1: Thong tin nha cho thue",
            "order": 4,
            "content_template": """DIEU 1: THONG TIN NHA CHO THUE

1.1. Dia chi: {{property_address}}
1.2. Dien tich: {{property_area}} m2
1.3. Ket cau: {{property_structure}}
1.4. Tinh trang nha: {{property_condition}}
1.5. Cac tien nghi kem theo: {{property_amenities}}
1.6. Giay to phap ly: So hong/So do so {{property_certificate}}""",
            "placeholders": [
                {"key": "property_address", "label": "Dia chi nha", "type": "string", "required": True},
                {"key": "property_area", "label": "Dien tich (m2)", "type": "number", "required": True},
                {"key": "property_structure", "label": "Ket cau (tang, phong)", "type": "string", "required": True},
                {"key": "property_condition", "label": "Tinh trang nha", "type": "string", "required": True},
                {"key": "property_amenities", "label": "Tien nghi kem theo", "type": "string", "required": True},
                {"key": "property_certificate", "label": "So giay chung nhan", "type": "string", "required": True}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_2",
            "section_name": "Dieu 2: Thoi han thue",
            "order": 5,
            "content_template": """DIEU 2: THOI HAN THUE

2.1. Thoi han thue: {{rental_period}}
2.2. Tu ngay: {{start_date}} den ngay: {{end_date}}
2.3. Gia han hop dong: Truoc khi het han {{renewal_notice}} ngay, neu Ben B co nhu cau gia han phai thong bao bang van ban cho Ben A.""",
            "placeholders": [
                {"key": "rental_period", "label": "Thoi han thue", "type": "string", "required": True, "default_value": "12 thang"},
                {"key": "start_date", "label": "Ngay bat dau", "type": "date", "required": True},
                {"key": "end_date", "label": "Ngay ket thuc", "type": "date", "required": True},
                {"key": "renewal_notice", "label": "Thong bao gia han truoc (ngay)", "type": "number", "required": True, "default_value": "30"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_3",
            "section_name": "Dieu 3: Gia thue va phuong thuc thanh toan",
            "order": 6,
            "content_template": """DIEU 3: GIA THUE VA PHUONG THUC THANH TOAN

3.1. Gia thue: {{rental_price}} dong/thang
3.2. Tien dat coc: {{deposit_amount}} dong (tuong duong {{deposit_months}} thang tien thue)
3.3. Ky thanh toan: {{payment_period}}
3.4. Phuong thuc thanh toan: {{payment_method}}
3.5. Han thanh toan: Truoc ngay {{payment_due}} hang thang
3.6. Dieu chinh gia thue: {{price_adjustment}}""",
            "placeholders": [
                {"key": "rental_price", "label": "Gia thue/thang (VND)", "type": "currency", "required": True},
                {"key": "deposit_amount", "label": "Tien dat coc (VND)", "type": "currency", "required": True},
                {"key": "deposit_months", "label": "So thang coc", "type": "number", "required": True, "default_value": "2"},
                {"key": "payment_period", "label": "Ky thanh toan", "type": "string", "required": True, "default_value": "Hang thang"},
                {"key": "payment_method", "label": "Phuong thuc thanh toan", "type": "string", "required": True, "default_value": "Tien mat hoac chuyen khoan"},
                {"key": "payment_due", "label": "Han thanh toan (ngay trong thang)", "type": "number", "required": True, "default_value": "5"},
                {"key": "price_adjustment", "label": "Quy dinh tang gia", "type": "string", "required": True, "default_value": "Co the dieu chinh sau 12 thang, thong bao truoc 30 ngay"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_4",
            "section_name": "Dieu 4: Cac chi phi khac",
            "order": 7,
            "content_template": """DIEU 4: CAC CHI PHI KHAC

4.1. Tien dien: Ben B thanh toan theo chi so cong to, don gia {{electricity_rate}} dong/kWh
4.2. Tien nuoc: Ben B thanh toan theo chi so dong ho, don gia {{water_rate}} dong/m3
4.3. Tien Internet: {{internet_fee}}
4.4. Phi quan ly (neu co): {{management_fee}}
4.5. Cac chi phi khac: {{other_fees}}""",
            "placeholders": [
                {"key": "electricity_rate", "label": "Gia dien (VND/kWh)", "type": "number", "required": True, "default_value": "3500"},
                {"key": "water_rate", "label": "Gia nuoc (VND/m3)", "type": "number", "required": True, "default_value": "15000"},
                {"key": "internet_fee", "label": "Phi internet", "type": "string", "required": True, "default_value": "Ben B tu dang ky va thanh toan"},
                {"key": "management_fee", "label": "Phi quan ly", "type": "string", "required": False, "default_value": "Khong co"},
                {"key": "other_fees", "label": "Chi phi khac", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_5",
            "section_name": "Dieu 5: Quyen va nghia vu cua cac ben",
            "order": 8,
            "content_template": """DIEU 5: QUYEN VA NGHIA VU CUA CAC BEN

5.1. Quyen va nghia vu cua Ben A:
a) Quyen:
- Nhan tien thue dung thoi han
- Kiem tra viec su dung nha dinh ky (bao truoc 24 gio)
- Don phuong cham dut hop dong neu Ben B vi pham nghiem trong

b) Nghia vu:
- Giao nha dung thoi han, dung tinh trang da thoa thuan
- Bao dam quyen su dung nha hop phap cho Ben B
- Sua chua nhung hu hong lon khong do loi Ben B

5.2. Quyen va nghia vu cua Ben B:
a) Quyen:
- Su dung nha theo muc dich da thoa thuan
- Yeu cau sua chua nhung hu hong lon

b) Nghia vu:
- Tra tien thue dung thoi han
- Giu gin, bao quan nha va trang thiet bi
- Khong tu y cai tao, sua doi ket cau nha
- Khong cho thue lai neu khong co su dong y cua Ben A
- Tra nha dung han trong tinh trang ban dau""",
            "placeholders": [],
            "is_optional": False
        },
        {
            "section_id": "article_6",
            "section_name": "Dieu 6: Cham dut hop dong",
            "order": 9,
            "content_template": """DIEU 6: CHAM DUT HOP DONG

6.1. Hop dong cham dut khi:
- Het thoi han thue
- Hai ben thoa thuan cham dut truoc han
- Ben B vi pham nghiem trong nghia vu hop dong

6.2. Cham dut truoc han:
- Ben B muon cham dut truoc han phai bao truoc {{early_termination_notice}} ngay
- Neu cham dut truoc han khong bao truoc: Mat tien coc

6.3. Xu ly tien coc:
- Neu Ben B tra nha dung han, khong hu hong: Hoan tra toan bo tien coc
- Neu co hu hong: Tru tien sua chua vao tien coc, hoan tra phan con lai""",
            "placeholders": [
                {"key": "early_termination_notice", "label": "Bao truoc cham dut (ngay)", "type": "number", "required": True, "default_value": "30"}
            ],
            "is_optional": False
        },
        {
            "section_id": "article_7",
            "section_name": "Dieu 7: Dieu khoan chung",
            "order": 10,
            "content_template": """DIEU 7: DIEU KHOAN CHUNG

7.1. Hop dong nay co hieu luc ke tu ngay ky.
7.2. Moi sua doi, bo sung phai duoc lap thanh van ban va co chu ky cua ca hai ben.
7.3. Khi phat sinh tranh chap, hai ben giai quyet bang thuong luong. Neu khong thanh, se giai quyet tai Toa an nhan dan co tham quyen.
7.4. Hop dong nay duoc lap thanh 02 ban co gia tri phap ly nhu nhau, moi ben giu 01 ban.

{{additional_terms}}""",
            "placeholders": [
                {"key": "additional_terms", "label": "Dieu khoan bo sung", "type": "string", "required": False}
            ],
            "is_optional": False
        },
        {
            "section_id": "signature",
            "section_name": "Chu ky",
            "order": 11,
            "content_template": """
BEN CHO THUE                                      BEN THUE
(Ky, ghi ro ho ten)                              (Ky, ghi ro ho ten)




{{party_a_name}}                                  {{party_b_name}}""",
            "placeholders": [],
            "is_optional": False
        }
    ],
    "rag_queries": [
        "hop dong thue nha o",
        "quyen va nghia vu ben thue",
        "dat coc thue nha",
        "cham dut hop dong thue"
    ],
    "legal_references": [
        "Bo luat Dan su 2015",
        "Luat Nha o 2014"
    ]
}


def seed_all_templates():
    """Seed all contract templates to MongoDB."""
    templates = [
        LABOR_CONTRACT_TEMPLATE,
        SALES_CONTRACT_TEMPLATE,
        RENTAL_CONTRACT_TEMPLATE
    ]

    for template in templates:
        try:
            save_template(template)
            print(f"Saved template: {template['template_id']} - {template['template_name']}")
        except Exception as e:
            print(f"Error saving template {template['template_id']}: {e}")

    print(f"\nSeeded {len(templates)} templates successfully!")


if __name__ == "__main__":
    seed_all_templates()
