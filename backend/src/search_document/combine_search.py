import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional

import redis

from search_document.search_with_legal_emb import QdrantSearch_legal
from search_document.search_elastic import search_data

logger = logging.getLogger(__name__)

# Read config from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
LEGAL_EMB_COLLECTION = os.getenv("LEGAL_EMB_COLLECTION", "law_with_legal_emb")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "legal_data_part2")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # 1 hour default

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


class SearchError(Exception):
    """Custom exception for search failures."""
    pass


class CombinedSearch:
    """
    Optimized hybrid search using:
    - paraphrase-vietnamese-law (semantic search) ~2GB VRAM
    - Elasticsearch BM25 (keyword search) 0GB VRAM
    - Reciprocal Rank Fusion (RRF) for combining results
    - Redis caching for performance

    Đã bỏ E5 và BGE-reranker để tiết kiệm ~7GB VRAM
    """

    # RRF constant (higher = more weight to lower-ranked docs)
    RRF_K = 60

    def __init__(self):
        self.legal_search = legal_search_instance
        self.elastic_index = elastic_params['index_name']
        self.elastic_top_k = elastic_params['top_k']

        # Initialize Redis for caching
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            self.redis.ping()
            self.cache_enabled = True
            logger.info(f"Redis cache enabled at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError:
            self.redis = None
            self.cache_enabled = False
            logger.warning("Redis unavailable, caching disabled")

    def search(self, query_text: str, top_k: int = 30) -> List[str]:
        """
        Perform combined search with RRF fusion and caching.

        Args:
            query_text: The query string.
            top_k: Number of top results to return.

        Returns:
            List of document texts, ranked by RRF score.

        Raises:
            SearchError: If all search services are unavailable.
        """
        # Check cache first
        cache_key = self._get_cache_key(query_text, top_k)
        cached_results = self._get_from_cache(cache_key)
        if cached_results is not None:
            logger.info(f"Cache hit for query: {query_text[:50]}...")
            return cached_results

        # Perform searches with error handling
        semantic_results = []
        elastic_results = []

        # Try semantic search (Qdrant)
        try:
            semantic_results = self.legal_search.search(query_text, limit=top_k)
            logger.debug(f"Semantic search returned {len(semantic_results)} results")
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # Try lexical search (Elasticsearch)
        try:
            elastic_results = search_data(
                self.elastic_index,
                query_text,
                top_k=self.elastic_top_k
            )
            logger.debug(f"Elasticsearch returned {len(elastic_results)} results")
        except Exception as e:
            logger.warning(f"Elasticsearch search failed: {e}")

        # Check if we have any results
        if not semantic_results and not elastic_results:
            raise SearchError("All search services unavailable")

        # Apply RRF fusion
        combined_results = self._rrf_fusion(
            semantic_results,
            elastic_results,
            top_k=top_k
        )

        # Cache results
        self._set_to_cache(cache_key, combined_results)

        logger.info(f"Search completed: {len(combined_results)} results for '{query_text[:50]}...'")
        return combined_results

    def _rrf_fusion(
        self,
        semantic_results: List[Any],
        elastic_results: List[Dict],
        top_k: int = 30
    ) -> List[str]:
        """
        Apply Reciprocal Rank Fusion to combine results from multiple sources.

        RRF score = sum(1 / (k + rank)) for each source where document appears.
        Documents appearing in both lists get higher scores.

        Args:
            semantic_results: Results from Qdrant (has .payload["text"])
            elastic_results: Results from Elasticsearch (has ["text"])
            top_k: Number of results to return

        Returns:
            List of document texts sorted by RRF score (highest first)
        """
        scores: Dict[int, float] = {}
        doc_map: Dict[int, str] = {}  # doc_id -> text

        # Score from semantic search
        for rank, hit in enumerate(semantic_results):
            try:
                text = hit.payload.get("text", "")
                if not text:
                    continue
                doc_id = hash(text)
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.RRF_K + rank + 1)
                doc_map[doc_id] = text
            except (AttributeError, KeyError) as e:
                logger.debug(f"Skip invalid semantic result: {e}")
                continue

        # Score from lexical search
        for rank, doc in enumerate(elastic_results):
            try:
                text = doc.get("text", "")
                if not text:
                    continue
                doc_id = hash(text)
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (self.RRF_K + rank + 1)
                doc_map[doc_id] = text
            except (AttributeError, KeyError) as e:
                logger.debug(f"Skip invalid elastic result: {e}")
                continue

        # Sort by RRF score (descending) and return top-k texts
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:top_k]]

    def _get_cache_key(self, query_text: str, top_k: int) -> str:
        """Generate cache key from query and parameters."""
        key_content = f"{query_text}:{top_k}"
        return f"search:{hashlib.md5(key_content.encode()).hexdigest()}"

    def _get_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """Get results from Redis cache."""
        if not self.cache_enabled or not self.redis:
            return None
        try:
            cached = self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.debug(f"Cache read error: {e}")
        return None

    def _set_to_cache(self, cache_key: str, results: List[str]) -> None:
        """Store results in Redis cache."""
        if not self.cache_enabled or not self.redis:
            return
        try:
            self.redis.setex(cache_key, CACHE_TTL, json.dumps(results, ensure_ascii=False))
        except redis.RedisError as e:
            logger.debug(f"Cache write error: {e}")

    def clear_cache(self, pattern: str = "search:*") -> int:
        """
        Clear search cache entries matching pattern.

        Args:
            pattern: Redis key pattern to match (default: all search keys)

        Returns:
            Number of keys deleted
        """
        if not self.cache_enabled or not self.redis:
            return 0
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
        except redis.RedisError as e:
            logger.warning(f"Cache clear error: {e}")
        return 0
