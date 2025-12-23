import torch
import logging
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class QdrantSearch_e5:
    def __init__(self, host: str, collection_name: str, model_name: str, use_fp16: bool = True):
        self.client = QdrantClient(host)
        self.collection_name = collection_name

        # Auto-detect GPU
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"E5 using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("E5 using CPU")

        self.model = SentenceTransformer(model_name, device=device)
        
    def encode_query(self, query_text: str):
        """Encode the query text into dense and sparse vectors"""
        query_text = "query: "+ query_text
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec

    def search(self, query_text: str, limit: int = 20):
        """Perform the search in Qdrant with the given query text and retrieve up to limit results"""
        dense_vec = self.encode_query(query_text)

        # E5 collection uses unnamed vectors, so query directly without prefetch/fusion
        results = self.client.query_points(
            self.collection_name,
            query=dense_vec.tolist(),
            with_payload=True,
            limit=limit,
        )

        return results

    
