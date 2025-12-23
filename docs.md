# Lựa chọn Model thay thế GPT-4o-mini

## Cấu hình máy hiện tại
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **VRAM khả dụng**: ~3-4GB (đang dùng ~8.7GB cho embedding models)

## Tình huống hiện tại

Hệ thống đang sử dụng **GPT-4o-mini** cho:
1. Intent routing (phân loại câu hỏi)
2. Query reflection (viết lại câu hỏi)
3. Phân tích hợp đồng (7 lần gọi LLM)
4. Trả lời câu hỏi pháp luật

## Các model thay thế tiềm năng

| Model | Params | VMLU Score | VRAM (4-bit) | Phù hợp RTX 3060 |
|-------|--------|------------|--------------|------------------|
| **SeaLLM-7B-v2.5** | 7B | 53.30% | ~5GB | Cần tắt embedding |
| **Vistral-7B-Chat** | 7B | 50.07% | ~5GB | Cần tắt embedding |
| **Qwen2.5-3B-Instruct** | 3B | - | ~2.5GB | Chạy được song song |
| **GPT-4o-mini** | - | - | 0GB | Tốn phí API |

## Khuyến nghị cho RTX 3060 12GB

### Phương án 1: Thay GPT-4o-mini bằng SeaLLM-7B-v2.5 (Khuyến nghị)

**Kiến trúc mới:**
```
[Query] → [Elasticsearch BM25] → [SeaLLM-7B rerank + generate]
```

**Thay đổi:**
- Tắt embedding models (BGE-M3, E5, paraphrase-vietnamese-law)
- Chỉ dùng Elasticsearch cho lexical search
- Dùng SeaLLM-7B cho cả rerank và generation
- VRAM: ~5GB (4-bit quantization)

**Ưu điểm:**
- SeaLLM-7B hiểu tiếng Việt tốt hơn GPT-4o-mini
- VMLU: 53.30% (vượt GPT-3.5)
- Không tốn phí API
- Self-hosted, bảo mật dữ liệu

### Phương án 2: Hybrid - Embedding + LLM nhỏ

**Kiến trúc:**
```
[Query] → [paraphrase-vietnamese-law] → [Qwen2.5-3B generate]
```

- Giữ 1 embedding model (~2GB)
- Dùng Qwen2.5-3B-Instruct (~2.5GB 4-bit)
- Tổng ~5GB VRAM

### Phương án 3: Giữ nguyên GPT-4o-mini
- Nếu GPT-4o-mini trả lời không chính xác, có thể do:
  - Prompt chưa tối ưu
  - Context retrieval không tốt
  - Thiếu hướng dẫn cụ thể cho domain pháp luật

## Lưu ý quan trọng

- `paraphrase-vietnamese-law` là **embedding model**, KHÔNG phải LLM
- Embedding model dùng cho **retrieval** (tìm kiếm văn bản)
- LLM dùng cho **generation** (sinh câu trả lời)

## Cấu hình embedding hiện tại

```bash
# Dùng paraphrase-vietnamese-law (đã cấu hình)
USE_LEGAL_EMB=true

# Quay lại BGE-M3
USE_LEGAL_EMB=false
```
