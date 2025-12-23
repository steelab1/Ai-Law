"""
Re-index existing Qdrant data with paraphrase-vietnamese-law embeddings.

This script:
1. Reads documents from existing BGE collection
2. Creates new collection with legal embeddings
3. Indexes all documents with the new model

Usage:
    python reindex_with_legal_emb.py
"""
import os
import sys
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
SOURCE_COLLECTION = os.getenv("BGE_COLLECTION", "law_with_bge_round1")
TARGET_COLLECTION = os.getenv("LEGAL_EMB_COLLECTION", "law_with_legal_emb")
MODEL_NAME = "minhquan6203/paraphrase-vietnamese-law"
VECTOR_SIZE = 768
BATCH_SIZE = 32


def get_all_documents(client: QdrantClient, collection_name: str, batch_size: int = 100):
    """
    Fetch all documents from a Qdrant collection.

    Args:
        client: Qdrant client
        collection_name: Source collection
        batch_size: Number of documents per batch

    Yields:
        Document payloads
    """
    # Get collection info
    collection_info = client.get_collection(collection_name)
    total_points = collection_info.points_count
    logger.info(f"Total documents in {collection_name}: {total_points}")

    offset = None
    fetched = 0

    while True:
        # Scroll through collection
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't need old vectors
        )

        if not records:
            break

        for record in records:
            yield record.payload
            fetched += 1

        offset = next_offset

        if offset is None:
            break

    logger.info(f"Fetched {fetched} documents from {collection_name}")


def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int):
    """Create collection if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists:
        logger.warning(f"Collection {collection_name} already exists. Deleting...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created collection: {collection_name}")


def reindex_documents(
    client: QdrantClient,
    source_collection: str,
    target_collection: str,
    model: SentenceTransformer,
    batch_size: int = 32
):
    """
    Re-index documents from source to target collection.

    Args:
        client: Qdrant client
        source_collection: Source collection name
        target_collection: Target collection name
        model: SentenceTransformer model for encoding
        batch_size: Batch size for encoding and upserting
    """
    # Get total count for progress bar
    collection_info = client.get_collection(source_collection)
    total_points = collection_info.points_count

    # Collect documents in batches
    batch = []
    point_id = 0

    with tqdm(total=total_points, desc="Re-indexing") as pbar:
        for doc in get_all_documents(client, source_collection, batch_size=100):
            batch.append(doc)

            if len(batch) >= batch_size:
                # Process batch
                texts = [d.get('text', '') for d in batch]

                # Encode with model
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                )

                # Create points
                points = []
                for doc, emb in zip(batch, embeddings):
                    point = PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload=doc
                    )
                    points.append(point)
                    point_id += 1

                # Upsert to target collection
                client.upsert(collection_name=target_collection, points=points)
                pbar.update(len(batch))
                batch = []

        # Process remaining documents
        if batch:
            texts = [d.get('text', '') for d in batch]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            points = []
            for doc, emb in zip(batch, embeddings):
                point = PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload=doc
                )
                points.append(point)
                point_id += 1

            client.upsert(collection_name=target_collection, points=points)
            pbar.update(len(batch))

    logger.info(f"Re-indexed {point_id} documents to {target_collection}")
    return point_id


def main():
    logger.info("=" * 60)
    logger.info("Re-indexing with paraphrase-vietnamese-law")
    logger.info("=" * 60)

    # Initialize Qdrant client
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}")
    client = QdrantClient(QDRANT_HOST)

    # Check source collection exists
    try:
        source_info = client.get_collection(SOURCE_COLLECTION)
        logger.info(f"Source collection: {SOURCE_COLLECTION} ({source_info.points_count} documents)")
    except Exception as e:
        logger.error(f"Source collection {SOURCE_COLLECTION} not found: {e}")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Create target collection
    create_collection_if_not_exists(client, TARGET_COLLECTION, VECTOR_SIZE)

    # Re-index
    count = reindex_documents(
        client=client,
        source_collection=SOURCE_COLLECTION,
        target_collection=TARGET_COLLECTION,
        model=model,
        batch_size=BATCH_SIZE
    )

    # Verify
    target_info = client.get_collection(TARGET_COLLECTION)
    logger.info(f"Target collection: {TARGET_COLLECTION} ({target_info.points_count} documents)")

    logger.info("=" * 60)
    logger.info("Re-indexing completed successfully!")
    logger.info("=" * 60)

    # Test search
    logger.info("\nTesting search...")
    test_query = "Thời gian thử việc tối đa là bao lâu?"
    query_vector = model.encode(test_query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=TARGET_COLLECTION,
        query_vector=query_vector,
        limit=3
    )

    logger.info(f"Query: {test_query}")
    for i, result in enumerate(results):
        logger.info(f"{i+1}. Score: {result.score:.4f} - {result.payload.get('text', '')[:100]}...")


if __name__ == "__main__":
    main()
