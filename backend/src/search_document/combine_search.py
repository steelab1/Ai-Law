import os
import logging
from search_document.search_with_legal_emb import QdrantSearch_legal
from search_document.search_elastic import search_data

logger = logging.getLogger(__name__)

# Read config from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
LEGAL_EMB_COLLECTION = os.getenv("LEGAL_EMB_COLLECTION", "law_with_legal_emb")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "legal_data_part2")

# Khởi tạo legal embedding search (paraphrase-vietnamese-law)
# Đã bỏ E5 và BGE-M3 để tiết kiệm VRAM (~7GB)
logger.info("Using paraphrase-vietnamese-law for legal embedding search (optimized)")
legal_search_instance = QdrantSearch_legal(
    host=QDRANT_HOST,
    collection_name=LEGAL_EMB_COLLECTION,
    model_name="minhquan6203/paraphrase-vietnamese-law"
)

elastic_params = {
    'index_name': ELASTIC_INDEX,
    'top_k': 30
}


class CombinedSearch:
    """
    Optimized search using:
    - paraphrase-vietnamese-law (semantic search) ~2GB VRAM
    - Elasticsearch BM25 (keyword search) 0GB VRAM

    Đã bỏ E5 và BGE-reranker để tiết kiệm ~7GB VRAM
    """

    def __init__(self):
        self.legal_search = legal_search_instance
        self.elastic_index = elastic_params['index_name']
        self.elastic_top_k = elastic_params['top_k']

    def search(self, query_text, top_k=30):
        """
        Perform combined search: Legal embedding + Elasticsearch.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to retrieve from each method.

        Returns:
            list: Combined search results (deduplicated).
        """
        combined_results = []

        # 1. Semantic search với paraphrase-vietnamese-law
        legal_results = self.legal_search.search(query_text, limit=top_k)
        for result in legal_results:
            combined_results.append(result.payload["text"])

        # 2. Keyword search với Elasticsearch BM25
        elastic_results = search_data(self.elastic_index, query_text, top_k=self.elastic_top_k)
        for result in elastic_results:
            combined_results.append(result['text'])

        # Deduplicate
        combined_results = list(set(combined_results))

        return combined_results
