"""
Import dataset vào Elasticsearch cho lexical search (BM25)
Chạy sau khi đã có Elasticsearch container running
"""
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# ===== CONFIG =====
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "legal_data_part2"
BATCH_SIZE = 500

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

# Tạo index với mapping tối ưu cho tiếng Việt
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
                "text": {
                    "type": "text",
                    "analyzer": "vietnamese_analyzer"
                },
                "question": {
                    "type": "text",
                    "analyzer": "vietnamese_analyzer"
                },
                "context": {
                    "type": "text",
                    "analyzer": "vietnamese_analyzer"
                },
                "cid": {
                    "type": "keyword"
                }
            }
        }
    }
)
print(f"Index '{INDEX_NAME}' created!")

# ===== Load dataset =====
print("\nLoading dataset...")
dataset = load_dataset("YuITC/Vietnamese-legal-documents")
print(f"Dataset loaded: {len(dataset['train'])} documents")

# ===== Import data =====
print("\nImporting data to Elasticsearch...")

def generate_actions():
    """Generator để bulk index"""
    for idx, item in enumerate(dataset['train']):
        # Lấy context
        context = item['context_list'][0] if item['context_list'] else ""

        yield {
            "_index": INDEX_NAME,
            "_id": item['qid'],
            "_source": {
                "text": item['question'],  # Field chính cho search
                "question": item['question'],
                "context": context,
                "cid": item['cid']
            }
        }

# Bulk index với progress bar
success, failed = 0, 0
for ok, result in tqdm(
    helpers.streaming_bulk(
        es,
        generate_actions(),
        chunk_size=BATCH_SIZE,
        raise_on_error=False
    ),
    total=len(dataset['train']),
    desc="Indexing"
):
    if ok:
        success += 1
    else:
        failed += 1

# Refresh index
es.indices.refresh(index=INDEX_NAME)

# ===== Done =====
doc_count = es.count(index=INDEX_NAME)['count']
print(f"\nDone!")
print(f"  Index: {INDEX_NAME}")
print(f"  Documents indexed: {doc_count}")
print(f"  Success: {success}, Failed: {failed}")
