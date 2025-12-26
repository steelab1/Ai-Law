import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# ===== CONFIG =====
MODEL_NAME = "minhquan6203/paraphrase-vietnamese-law"
COLLECTION_NAME = "vietnamese_legal_docs"
VECTOR_SIZE = 768
BATCH_SIZE = 32

# ===== BƯỚC 0: Kiểm tra GPU =====
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ===== BƯỚC 1: Tải dataset =====
print("\nLoading dataset...")
dataset = load_dataset("YuITC/Vietnamese-legal-documents")
print(f"Dataset loaded: {len(dataset['train'])} documents")

# ===== BƯỚC 2: Load mô hình paraphrase-vietnamese-law =====
print(f"\nLoading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device)
print("Model loaded!")

# ===== BƯỚC 3: Hàm tạo embeddings =====
def create_embeddings(texts, batch_size=BATCH_SIZE):
    """Tạo embeddings với SentenceTransformer"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # Chuẩn hóa cho cosine similarity
        show_progress_bar=False
    )
    return embeddings

# ===== BƯỚC 4: Tạo Qdrant collection =====
print("\nConnecting to Qdrant Docker...")
client = QdrantClient(host="localhost", port=6333)

# Xóa collection cũ nếu tồn tại, rồi tạo mới
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}...")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE,
        distance=models.Distance.COSINE
    )
)
print(f"Collection '{COLLECTION_NAME}' created!")

# ===== BƯỚC 5: Upload dữ liệu =====
print("\nUploading data to Qdrant...")
total_uploaded = 0

for batch_idx, batch in enumerate(dataset['train'].iter(batch_size=BATCH_SIZE)):
    # Tạo embeddings từ question
    question_vectors = create_embeddings(batch['question'])

    # Chuẩn bị points
    points = []
    for i, qid in enumerate(batch['qid']):
        # Lấy context
        context = batch['context_list'][i][0] if batch['context_list'][i] else ""

        points.append(
            models.PointStruct(
                id=int(qid),
                vector=question_vectors[i].tolist(),
                payload={
                    "text": batch['question'][i],  # Field chính cho search
                    "question": batch['question'][i],
                    "cid": batch['cid'][i],
                    "context": context
                }
            )
        )

    # Upsert vào Qdrant
    client.upsert(COLLECTION_NAME, points)
    total_uploaded += len(points)

    # Log progress và dọn dẹp memory
    if (batch_idx + 1) % 10 == 0:
        print(f"Processed {total_uploaded} documents...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ===== DONE =====
final_count = client.count(COLLECTION_NAME).count
print(f"\nDone! Collection '{COLLECTION_NAME}': {final_count} documents")
print(f"Model used: {MODEL_NAME}")
