import os
from search_document.search_with_bge import QdrantSearch_bge
from search_document.search_with_e5 import QdrantSearch_e5
from search_document.search_elastic import search_data

# Read Qdrant config from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
BGE_COLLECTION = os.getenv("BGE_COLLECTION", "law_with_bge_round1")
E5_COLLECTION = os.getenv("E5_COLLECTION", "law_with_e5_emb_not_finetune")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "legal_data_part2")

# Khởi tạo các search class ở cấp module để tái sử dụng
bge_search_instance = QdrantSearch_bge(
        host=QDRANT_HOST,
        collection_name=BGE_COLLECTION,
        model_name="BAAI/bge-m3",
        use_fp16=True
    )

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
        self.bge_search = bge_search_instance  # Reuse instance
        self.e5_search = e5_search_instance  # Reuse instance
        self.elastic_index = elastic_params['index_name']
        self.elastic_top_k = elastic_params['top_k']

    def search(self, query_text, top_k=30):
        """
        Perform a combined search across BGE, E5, and Elasticsearch.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to retrieve from each method.

        Returns:
            list: Combined search results.
        """
        # Perform searches
        bge_results = self.bge_search.search(query_text, limit=top_k)
        e5_results = self.e5_search.search(query_text, limit=top_k)
        elastic_results = search_data(self.elastic_index, query_text, top_k=self.elastic_top_k)

        # Combine and normalize results
        combined_results = []

        # Process BGE results
        for result in bge_results.points:
            combined_results.append(result.payload["text"])

        # Process E5 results
        for result in e5_results.points:
            combined_results.append(result.payload["text"])

        # Process Elasticsearch results
        for result in elastic_results:
            combined_results.append(result['text'])

        combined_results = list(set(combined_results))

        return combined_results
