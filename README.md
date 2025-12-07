
# LLM chat bot - Hệ thống Hỏi đáp trên tài liệu pháp luật Việt Nam
Trong dự án này, tôi xây dựng một chatbot Hỏi - Đáp hoàn chỉnh về tài liệu pháp luật Việt Nam.

Link bộ dữ liệu: [Link dữ liệu](https://drive.google.com/drive/folders/1HyF8-EfL4w0G3spBbhcc0jTOqdc4XUhB)

# Mục lục

<!--ts-->
   * [Cấu trúc dự án](#cấu-trúc-dự-án)
   * [Bắt đầu](#bắt-đầu)
      * [Chuẩn bị môi trường](#chuẩn-bị-môi-trường)
      * [Chạy ứng dụng bằng docker container trên local](#chạy-ứng-dụng-docker-container-trên-local)
   * [Các dịch vụ của ứng dụng](#các-dịch-vụ-của-ứng-dụng)
      * [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
        * [Tổng quan hệ thống](#tổng-quan-hệ-thống)
        * [Xây dựng Vector Database và Elasticsearch](#xây-dựng-vectordb-và-elasticsearch)
        * [Luồng RAG trả lời](#luồng-rag-trả-lời)
        * [Tinh chỉnh mô hình rerank](#tinh-chỉnh-mô-hình-rerank)
        * [Tinh chỉnh LLM cho bước sinh câu trả lời](#tinh-chỉnh-llm-cho-bước-sinh-câu-trả-lời)
   * [Demo](#demo)
<!--te-->

# Cấu trúc dự án
```bash
├── backend                                   
│   ├── requirements.txt                        # các phụ thuộc cho backend 
│   ├── entrypoint.sh                           # script chạy backend  
│   ├── src                                     # Mã nguồn backend
│   │   ├── search_document                             
│   │   │   ├── combine_search.py              # trộn kết quả từ Bge-m3, e5
│   │   │   ├── reranking.py                   # reranking 
│   │   │   ├── search_elastic.py              # tìm kiếm bằng elasticsearch
│   │   │   ├── search_with_bge.py             # tìm kiếm bằng Bge-m3
│   │   │   └── search_with_e5.py              # tìm kiếm bằng Multilingual-e5-large
│   │   ├── agent.py                            # thử nghiệm react agent với tool tìm kiếm
│   │   ├── app.py                              # Entry point của FastAPI backend
│   │   ├── brain.py                            # Logic ra quyết định với OpenAI client
│   │   ├── cache.py                            # Cache của ứng dụng
│   │   ├── celery_app.py                       # Cấu hình hàng đợi tác vụ Celery
│   │   ├── config.py                           # File cấu hình backend
│   │   ├── database.py                         # Kết nối cơ sở dữ liệu
│   │   ├── models.py                           # Các model của cơ sở dữ liệu
│   │   ├── tavily_search.py                    # định nghĩa tool tìm kiếm internet
│   │   ├── schemas.py                          # Các schema dữ liệu cho API
│   │   ├── task.py                             # Định nghĩa task cho celery
│   │   └── utils.py                            # Các hàm tiện ích cho backend                  
├── chatbot-ui                                  # Ứng dụng frontend chatbot
│   ├── chat_interface.py                       # Logic giao diện chatbot
│   ├── config.toml                             # File cấu hình cho chatbot                  
│   ├── entrypoint.sh                           # Script khởi chạy chatbot
│   ├── requirements.txt                        # Các phụ thuộc Python cho chatbot
├── finetune_llm                                # Thư mục tinh chỉnh LLM
│   ├── download_model.py                       # tải model gốc          
│   ├── finetune.py                             # tinh chỉnh LLM cho sinh câu trả lời
│   ├── gen_data.py                             # Code tạo dữ liệu             
│   ├── merge_with_base.py                      # Gộp trọng số tinh chỉnh với model gốc
│   └── pdf                                     
├── images                                      # Thư mục lưu ảnh
├── retrieval                                   # Thư mục Retrieval
│   ├── FlagEmbedding                           # mã nguồn tinh chỉnh
│   ├── hard_negative_bge_round1.py             # tìm kiếm bằng bge-m3
│   ├── hard_negative_e5.py                     # tìm kiếm bằng e5
│   ├── create_data_rerank.py                   # tạo dữ liệu cho reranking   
│   ├── finetune.sh                             # Script tinh chỉnh bge-reranker-v2-m3
│   └── setup_env.sh                            # Script tạo môi trường
```
# Bắt đầu

Để bắt đầu với dự án, thực hiện các bước sau

## Chuẩn bị môi trường
Cài đặt toàn bộ phụ thuộc của dự án trên máy local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r chatbot-ui/requirements.txt
```
Khởi động ứng dụng:
```bash
sh backend/entrypoint.sh
sh chatbot-ui/entrypoint.sh
```

# Các dịch vụ của ứng dụng 

## RAG (Retrieval-Augmented Generation) 

### Tổng quan hệ thống

![rag_system](images/rag_flow.jpg)

### Xây dựng VectorDB và Elasticsearch 
- Vì độ dài của mỗi quy định khá lớn nên bước đầu tiên cần chia nhỏ (chunk) tài liệu. Các chunk này được đưa qua 2 mô hình embedding Bge-m3 và Multilingual-e5-large, sau đó vector embedding được lưu vào Qdrant.
- Đồng thời, các quy định cũng được lưu vào Elasticsearch để tăng độ chính xác của truy xuất dựa trên khớp từ vựng.

### Luồng RAG trả lời

- **Định tuyến ý định người dùng**: Dựa trên câu hỏi hiện tại và lịch sử trò chuyện để xác định người dùng đang chitchat hay hỏi về pháp luật. Mô hình `gpt-4o-mini` kết hợp `few-shot prompting` được dùng cho bước này. Nếu là chitchat, gọi OpenAI để trả lời cuối cùng; nếu không, chuyển sang bước rewrite truy vấn.

- **Query reflection**: Lịch sử trò chuyện và câu hỏi hiện tại được viết lại thành một câu đơn đầy đủ nghĩa để truy xuất dễ hơn. Mô hình dùng: `gpt-4o-mini`.

- **Truy xuất tài liệu liên quan**: Truy vấn đã viết lại được đưa qua hai mô hình embedding Bge-m3 và Multilingual-e5-large, Qdrant dùng để truy xuất các tài liệu liên quan ngữ nghĩa. Elasticsearch cũng được dùng cho truy xuất dựa trên từ vựng. Các tài liệu lấy được được gộp lại và loại bỏ trùng lặp để hạn chế mất thông tin.

- **Reranking**: Nếu không rerank, số tài liệu trả về lớn; đưa hết vào ChatModel (OpenAI) có thể vượt giới hạn token và tốn chi phí. Nếu giảm top_k quá nhỏ sẽ bỏ sót thông tin. Do đó top_k tài liệu từ bước trước được đưa qua mô hình rerank để sắp xếp lại, lấy top5 điểm cao nhất.

- **Sinh câu trả lời cuối**: LLM kết hợp top5 tài liệu sau rerank với câu hỏi và lịch sử trò chuyện để tạo phản hồi. Trong prompt, LLM được yêu cầu trả về 'no' nếu tài liệu không chứa câu trả lời; nếu khác 'no' thì đó là đáp án cuối. Nếu là 'no' sẽ gọi tool tìm kiếm ở bước sau.

- **Gọi công cụ và sinh tiếp**: Dùng công cụ tìm kiếm Tavily để tìm thông tin trên internet, sau đó đưa nội dung này vào LLM để sinh câu trả lời.

### Tinh chỉnh mô hình rerank
Tạo môi trường 
```bash
cd retrieval
sh setup_env.sh
```
#### Tạo dữ liệu tinh chỉnh
- Dữ liệu huấn luyện là file json, mỗi dòng là một dict:

```shell
{"query": str, "pos": List[str], "neg":List[str]}
```
`query` là truy vấn, `pos` là danh sách văn bản dương tính, `neg` là danh sách văn bản âm tính. 
- Với mỗi mô hình embedding: lấy 25 chunk có độ tương đồng cao nhất cho mỗi truy vấn. Nếu chunk nằm trong dữ liệu đã gán nhãn thì là positive, ngược lại là negative; kết quả của các mô hình embedding được tổng hợp lại.

- Thực hiện các bước sau để tạo tập huấn luyện

```bash
Step1: cd retrieval
Step2: CUDA_VISIBLE_DEVICES=0 python create_data_rerank.py
```

#### Tinh chỉnh BGE-v2-m3
Tinh chỉnh BGE-v2-m3 với các tham số: 

    - epochs: 6
    - learning_rate: 1e-5
    - batch_size = 2

Chạy script huấn luyện
```bash
sh finetune.sh
```
### Tinh chỉnh LLM cho bước sinh câu trả lời
#### Tạo + định dạng dữ liệu huấn luyện
- Dữ liệu huấn luyện ở dạng hội thoại.
```shell
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
- Các bước tạo tập huấn luyện + kiểm thử:
```bash
Step1: cd finetune_llm
Step2: python gen_data.py
```
- Số lượng mẫu huấn luyện: 10.000; số mẫu kiểm thử: 1.000

#### Tinh chỉnh LLM
- Model gốc dùng để tinh chỉnh: [1TuanPham/T-VisStar-7B-v0.1](https://huggingface.co/1TuanPham/T-VisStar-7B-v0.1). Model này đạt thứ hạng cao trên VMLU Leaderboard của các model fine-tune.
- Sử dụng [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) từ thư viện trl để tinh chỉnh. Đồng thời áp dụng kỹ thuật [QLora](https://arxiv.org/abs/2305.14314) để giảm bộ nhớ khi tinh chỉnh bằng cách lượng tử hóa nhưng vẫn giữ hiệu năng.

Chạy script huấn luyện
```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py
```
- Kết quả huấn luyện trên WanDB:
![Tracking training](images/tracking_finetune_llm.png)

- Gộp trọng số tinh chỉnh với model gốc
Chạy script gộp
```bash
python merge_with_base.py
```

### Đánh giá 

Các metric đánh giá đang dùng:

- **Recall@k**: Đo mức độ truy xuất đúng thông tin
- **Correctness**: Đánh giá câu trả lời sinh ra so với đáp án tham chiếu.

Tập golden dùng để đánh giá gồm 1000 mẫu, mỗi mẫu gồm 3 trường: query, related_documents, answer.


**Recall@k**
|Model               | K=3    | K =5   | K=10    |
|-----------------   |--------|--------|---------|
|BGE-m3              | 55.11% | 63.43% | 72.18%  |
|E5                  | 54.61% | 63.53% | 72.02%  |
|Elasticsearch       | 42.54% | 49.61% | 56.85%  |
|Ensemble            | 68.38% | 74.85% | 80.66%  |
|Ensemble + rerank   | 79.82% | 82,82% | 87.66%  |

**Correctness**

Điểm được đánh giá trên thang 5 và đạt độ chính xác 4.27/5
# DEMO       
![demo](images/demo.png)
