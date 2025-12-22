# Contract Drafting & Analysis Module

## Tong quan

Module nay cung cap 2 chuc nang chinh:

1. **Soan thao hop dong** - Tao hop dong tu template co san trong MongoDB
2. **Phan tich hop dong** - Upload file PDF/Word, phan tich va so sanh voi mau chuan

## Cau truc thu muc

```
backend/src/
├── contract_drafting/           # Soan thao hop dong
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   ├── models.py               # MongoDB operations
│   ├── prompts.py              # LLM prompts
│   ├── generator.py            # Core logic
│   └── seed_templates.py       # Script seed templates
│
├── contract_analysis/          # Phan tich hop dong
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   ├── models.py               # MongoDB operations
│   ├── prompts.py              # LLM prompts
│   ├── extractor.py            # PDF/Word extraction
│   └── comparator.py           # Core logic
```

## Cai dat

### 1. Cai dat dependencies moi

```bash
pip install PyMuPDF python-docx python-multipart httpx
```

### 2. Seed contract templates vao MongoDB

```bash
cd backend/src
python -m contract_drafting.seed_templates
```

## API Endpoints

### Contract Drafting

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| GET | `/contract/templates` | Danh sach templates |
| GET | `/contract/templates/{id}` | Chi tiet template + placeholders |
| POST | `/contract/draft` | Tao hop dong tu template |
| GET | `/contract/drafts/{user_id}` | Lich su drafts cua user |

### Contract Analysis

| Method | Endpoint | Mo ta |
|--------|----------|-------|
| POST | `/contract/analyze` | Upload file + phan tich |
| GET | `/contract/analysis/{id}` | Lay ket qua da luu |
| GET | `/contract/analyses/{user_id}` | Lich su phan tich |

## Su dung

### 1. Lay danh sach templates

```bash
curl http://localhost:8002/contract/templates
```

### 2. Lay chi tiet template

```bash
curl http://localhost:8002/contract/templates/tpl_labor_001
```

### 3. Tao hop dong

```bash
curl -X POST http://localhost:8002/contract/draft \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "template_id": "tpl_labor_001",
    "values": {
      "contract_number": "HD-2024-001",
      "contract_date": "01/01/2024",
      "party_a_name": "Cong ty ABC",
      "party_b_name": "Nguyen Van A"
    }
  }'
```

### 4. Phan tich hop dong

```bash
curl -X POST http://localhost:8002/contract/analyze \
  -F "file=@hopdong.pdf" \
  -F "user_id=user123"
```

## Loai hop dong ho tro

| Type | Ten | Templates |
|------|-----|-----------|
| `labor` | Hop dong lao dong | tpl_labor_001 |
| `sales` | Hop dong mua ban | tpl_sales_001 |
| `rental` | Hop dong thue | tpl_rental_001 |
| `services` | Hop dong dich vu | (coming soon) |
| `loan` | Hop dong vay | (coming soon) |
| `partnership` | Hop dong hop tac | (coming soon) |

## Phan tich hop dong bao gom

1. **Dieu khoan bat loi** - Tim cac dieu khoan gai bay, bat loi cho 1 ben
2. **So sanh mau chuan** - So sanh voi template, tim diem thieu
3. **Kiem tra tuan thu** - Kiem tra tuan thu phap luat VN
4. **Tom tat quyen/nghia vu** - Liet ke quyen va nghia vu cac ben
5. **Danh gia rui ro** - Cho diem rui ro tong the (0-10)

## Swagger UI

Truy cap http://localhost:8002/docs de xem API documentation.
