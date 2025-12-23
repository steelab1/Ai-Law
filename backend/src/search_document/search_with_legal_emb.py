"""
Vietnamese Legal Embedding Search
Using paraphrase-vietnamese-law model for better legal domain retrieval
"""
import torch
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


class QdrantSearch_legal:
    """
    Search using paraphrase-vietnamese-law embedding model.
    This model is specifically trained on Vietnamese legal documents
    and provides better semantic understanding for legal queries.
    """

    def __init__(self, host: str, collection_name: str, model_name: str = "minhquan6203/paraphrase-vietnamese-law"):
        self.client = QdrantClient(host)
        self.collection_name = collection_name
        self.vector_size = 768  # paraphrase-vietnamese-law output dimension

        # Auto-detect GPU
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Legal Embedding using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Legal Embedding using CPU")

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded legal embedding model: {model_name}")

    def encode_query(self, query_text: str):
        """Encode the query text into dense vector"""
        # Encode with normalization for cosine similarity
        embedding = self.model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def search(self, query_text: str, limit: int = 20):
        """
        Perform semantic search in Qdrant.

        Args:
            query_text: The query string
            limit: Number of top results to retrieve

        Returns:
            Qdrant search results
        """
        query_vector = self.encode_query(query_text)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        return results

    def search_with_score_threshold(self, query_text: str, limit: int = 20, score_threshold: float = 0.5):
        """
        Search with minimum score threshold.

        Args:
            query_text: The query string
            limit: Number of top results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            Filtered search results
        """
        query_vector = self.encode_query(query_text)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )

        return results


def create_legal_collection(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """
    Create a new Qdrant collection for legal embeddings.

    Args:
        client: Qdrant client instance
        collection_name: Name for the new collection
        vector_size: Embedding dimension (768 for paraphrase-vietnamese-law)
    """
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists:
        logger.info(f"Collection {collection_name} already exists")
        return False

    # Create collection with cosine similarity
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

    logger.info(f"Created collection {collection_name} with vector size {vector_size}")
    return True


def index_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list,
    model: SentenceTransformer,
    batch_size: int = 32
):
    """
    Index documents into Qdrant collection.

    Args:
        client: Qdrant client
        collection_name: Target collection
        documents: List of dicts with 'text' and optional 'metadata'
        model: SentenceTransformer model
        batch_size: Batch size for encoding
    """
    from qdrant_client.models import PointStruct

    total = len(documents)
    logger.info(f"Indexing {total} documents to {collection_name}")

    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc['text'] for doc in batch]

        # Encode batch
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Create points
        points = []
        for j, (doc, emb) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=i + j,
                vector=emb.tolist(),
                payload={"text": doc['text'], **doc.get('metadata', {})}
            )
            points.append(point)

        # Upsert batch
        client.upsert(collection_name=collection_name, points=points)

        logger.info(f"Indexed {min(i+batch_size, total)}/{total} documents")

    logger.info(f"Completed indexing {total} documents")


if __name__ == "__main__":
    # Test the search
    import os

    QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")

    searcher = QdrantSearch_legal(
        host=QDRANT_HOST,
        collection_name="law_with_legal_emb"  # New collection name
    )

    query = "Thời gian thử việc tối đa là bao lâu?"
    results = searcher.search(query, limit=5)

    print(f"Query: {query}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.score:.4f}")
        print(f"   Text: {result.payload.get('text', '')[:200]}...")
