import os
import logging
from search_document.search_with_bge import QdrantSearch_bge
from search_document.search_with_e5 import QdrantSearch_e5
from search_document.search_with_legal_emb import QdrantSearch_legal
from search_document.search_elastic import search_data

logger = logging.getLogger(__name__)

# Read Qdrant config from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
BGE_COLLECTION = os.getenv("BGE_COLLECTION", "law_with_bge_round1")
E5_COLLECTION = os.getenv("E5_COLLECTION", "law_with_e5_emb_not_finetune")
LEGAL_EMB_COLLECTION = os.getenv("LEGAL_EMB_COLLECTION", "law_with_legal_emb")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "legal_data_part2")

# Flag to use new legal embedding model instead of BGE-M3
USE_LEGAL_EMB = os.getenv("USE_LEGAL_EMB", "false").lower() == "true"

# Khởi tạo các search class ở cấp module để tái sử dụng
if USE_LEGAL_EMB:
    # Use paraphrase-vietnamese-law for better legal domain retrieval
    logger.info("Using paraphrase-vietnamese-law for legal embedding search")
    legal_search_instance = QdrantSearch_legal(
        host=QDRANT_HOST,
        collection_name=LEGAL_EMB_COLLECTION,
        model_name="minhquan6203/paraphrase-vietnamese-law"
    )
    bge_search_instance = None
else:
    # Use BGE-M3 (default)
    logger.info("Using BGE-M3 for embedding search")
    bge_search_instance = QdrantSearch_bge(
        host=QDRANT_HOST,
        collection_name=BGE_COLLECTION,
        model_name="BAAI/bge-m3",
        use_fp16=True
    )
    legal_search_instance = None

e5_search_instance = QdrantSearch_e5(
        host=QDRANT_HOST,
        collection_name=E5_COLLECTION,
        model_name="intfloat/multilingual-e5-large",
        use_fp16=True
    )

elastic_params = {
        'index_name': ELASTIC_INDEX,
        'top_k': 30
    }


class CombinedSearch:
    def __init__(self):
        self.use_legal_emb = USE_LEGAL_EMB
        if self.use_legal_emb:
            self.legal_search = legal_search_instance
            self.bge_search = None
        else:
            self.bge_search = bge_search_instance
            self.legal_search = None
        self.e5_search = e5_search_instance
        self.elastic_index = elastic_params['index_name']
        self.elastic_top_k = elastic_params['top_k']

    def search(self, query_text, top_k=30):
        """
        Perform a combined search across Legal/BGE, E5, and Elasticsearch.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to retrieve from each method.

        Returns:
            list: Combined search results.
        """
        combined_results = []

        # Perform primary embedding search (Legal or BGE)
        if self.use_legal_emb and self.legal_search:
            # Use paraphrase-vietnamese-law
            legal_results = self.legal_search.search(query_text, limit=top_k)
            for result in legal_results:
                combined_results.append(result.payload["text"])
        elif self.bge_search:
            # Use BGE-M3
            bge_results = self.bge_search.search(query_text, limit=top_k)
            for result in bge_results.points:
                combined_results.append(result.payload["text"])

        # E5 search
        e5_results = self.e5_search.search(query_text, limit=top_k)
        for result in e5_results.points:
            combined_results.append(result.payload["text"])

        # Elasticsearch lexical search
        elastic_results = search_data(self.elastic_index, query_text, top_k=self.elastic_top_k)
        for result in elastic_results:
            combined_results.append(result['text'])

        # Deduplicate
        combined_results = list(set(combined_results))

        return combined_results
