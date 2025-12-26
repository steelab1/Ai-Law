"""
Import vietnamese-legal-corpus-20k-raw vào Elasticsearch
Dataset: https://huggingface.co/datasets/52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw
"""
import re
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# ===== CONFIG =====
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "legal_data_part2"
BATCH_SIZE = 500

# Chunking config (same as import.py)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# ===== Hàm chunking =====
def clean_text(text):
    """Làm sạch text"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chia text thành các chunks với overlap"""
    if not text or len(text) < 100:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > chunk_size // 2:
                chunk = chunk[:cut_point + 1]
                end = start + cut_point + 1

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap
        if start >= len(text):
            break

    return chunks

# ===== Kết nối Elasticsearch =====
print(f"Connecting to Elasticsearch at {ELASTICSEARCH_URL}...")
es = Elasticsearch([ELASTICSEARCH_URL])

if not es.ping():
    print("ERROR: Cannot connect to Elasticsearch!")
    print("Make sure Elasticsearch is running: docker-compose up -d elasticsearch")
    exit(1)

print("Connected to Elasticsearch!")

# ===== Xóa index cũ và tạo mới =====
if es.indices.exists(index=INDEX_NAME):
    print(f"Deleting existing index: {INDEX_NAME}...")
    es.indices.delete(index=INDEX_NAME)

print(f"Creating index: {INDEX_NAME}...")
es.indices.create(
    index=INDEX_NAME,
    body={
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "vietnamese_analyzer": {
                        "type": "standard",
                        "stopwords": "_none_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "vietnamese_analyzer"},
                "title": {"type": "text", "analyzer": "vietnamese_analyzer"},
                "official_number": {"type": "keyword"},
                "document_type": {"type": "keyword"},
                "document_field": {"type": "keyword"},
                "issued_date": {"type": "keyword"},
                "effective_date": {"type": "keyword"},
                "place_issue": {"type": "keyword"},
                "signer": {"type": "text"},
                "url": {"type": "keyword"},
                "source_id": {"type": "integer"},
                "chunk_idx": {"type": "integer"},
                "total_chunks": {"type": "integer"}
            }
        }
    }
)
print(f"Index '{INDEX_NAME}' created!")

# ===== Load dataset =====
print("\nLoading dataset: vietnamese-legal-corpus-20k-raw...")
dataset = load_dataset("52100303-TranPhuocSang/vietnamese-legal-corpus-20k-raw")
print(f"Dataset loaded: {len(dataset['train'])} documents")

# ===== Chunking =====
print("\nChunking documents...")
all_chunks = []
for idx, doc in enumerate(tqdm(dataset['train'], desc="Chunking")):
    full_text = clean_text(doc.get('full_text', ''))
    title = clean_text(doc.get('title', ''))

    metadata = {
        'title': title,
        'official_number': doc.get('official_number', ''),
        'document_type': doc.get('document_type', ''),
        'document_field': doc.get('document_field', ''),
        'issued_date': doc.get('issued_date', ''),
        'effective_date': doc.get('effective_date', ''),
        'place_issue': doc.get('place_issue', ''),
        'signer': doc.get('signer', ''),
        'url': doc.get('url', ''),
        'source_id': doc.get('source_id', idx),
    }

    if full_text:
        chunks = chunk_text(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'chunk_idx': chunk_idx,
                'total_chunks': len(chunks),
                **metadata
            })
    elif title:
        all_chunks.append({
            'text': title,
            'chunk_idx': 0,
            'total_chunks': 1,
            **metadata
        })

print(f"Total chunks: {len(all_chunks)}")

# ===== Import vào Elasticsearch =====
print("\nImporting to Elasticsearch...")

def generate_actions():
    for idx, chunk in enumerate(all_chunks):
        yield {
            "_index": INDEX_NAME,
            "_id": idx,
            "_source": chunk
        }

success, failed = 0, 0
for ok, result in tqdm(
    helpers.streaming_bulk(
        es,
        generate_actions(),
        chunk_size=BATCH_SIZE,
        raise_on_error=False
    ),
    total=len(all_chunks),
    desc="Indexing"
):
    if ok:
        success += 1
    else:
        failed += 1

es.indices.refresh(index=INDEX_NAME)

# ===== Done =====
doc_count = es.count(index=INDEX_NAME)['count']
print(f"\n{'='*50}")
print(f"DONE!")
print(f"Index: {INDEX_NAME}")
print(f"Documents indexed: {doc_count}")
print(f"Success: {success}, Failed: {failed}")
print(f"{'='*50}")
