#!/usr/bin/env python3
"""
Import data from corpus.csv into Qdrant and Elasticsearch.

Usage:
    python import_data.py --corpus data/corpus.csv                    # Import 10% (default)
    python import_data.py --corpus data/corpus.csv --percent 100      # Import 100%
    python import_data.py --corpus data/corpus.csv --resume           # Resume from checkpoint
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from elasticsearch import Elasticsearch, helpers
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scripts/import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
BGE_COLLECTION = os.getenv("BGE_COLLECTION", "law_with_bge_round1")
E5_COLLECTION = os.getenv("E5_COLLECTION", "law_with_e5_emb_not_finetune")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "legal_data_part2")

BATCH_SIZE = 32  # Optimized for RTX 3060 (12GB VRAM)
CHECKPOINT_INTERVAL = 10000
CHECKPOINT_FILE = "scripts/.import_checkpoint.json"


class DataImporter:
    def __init__(self, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize clients
        self.qdrant = None
        self.es = None
        self.bge_model = None
        self.e5_model = None

    def connect(self):
        """Connect to Qdrant and Elasticsearch."""
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}")
        self.qdrant = QdrantClient(url=QDRANT_HOST)

        logger.info(f"Connecting to Elasticsearch at {ELASTICSEARCH_URL}")
        self.es = Elasticsearch(
            [ELASTICSEARCH_URL],
            verify_certs=False,
            ssl_show_warn=False
        )

        # Test connection by getting cluster info
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch cluster: {info['cluster_name']}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Elasticsearch: {e}")
        logger.info("Connected to both databases successfully")

    def load_models(self):
        """Load embedding models."""
        logger.info("Loading BGE-m3 model...")
        self.bge_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=True if self.device == "cuda" else False
        )

        logger.info("Loading E5-large model...")
        self.e5_model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            device=self.device
        )
        logger.info("Models loaded successfully")

    def create_collections(self, recreate: bool = False):
        """Create Qdrant collections and Elasticsearch index."""
        # BGE-m3 collection (dense + sparse)
        if recreate or not self.qdrant.collection_exists(BGE_COLLECTION):
            logger.info(f"Creating Qdrant collection: {BGE_COLLECTION}")
            if self.qdrant.collection_exists(BGE_COLLECTION):
                self.qdrant.delete_collection(BGE_COLLECTION)

            self.qdrant.create_collection(
                collection_name=BGE_COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
        else:
            logger.info(f"Collection {BGE_COLLECTION} already exists")

        # E5 collection (dense only)
        if recreate or not self.qdrant.collection_exists(E5_COLLECTION):
            logger.info(f"Creating Qdrant collection: {E5_COLLECTION}")
            if self.qdrant.collection_exists(E5_COLLECTION):
                self.qdrant.delete_collection(E5_COLLECTION)

            self.qdrant.create_collection(
                collection_name=E5_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Collection {E5_COLLECTION} already exists")

        # Elasticsearch index
        if recreate or not self.es.indices.exists(index=ELASTIC_INDEX):
            logger.info(f"Creating Elasticsearch index: {ELASTIC_INDEX}")
            if self.es.indices.exists(index=ELASTIC_INDEX):
                self.es.indices.delete(index=ELASTIC_INDEX)

            self.es.indices.create(
                index=ELASTIC_INDEX,
                body={
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "cid": {"type": "integer"}
                        }
                    }
                }
            )
        else:
            logger.info(f"Index {ELASTIC_INDEX} already exists")

    def encode_bge(self, texts: List[str]) -> Tuple[List, List]:
        """Encode texts with BGE-m3 model."""
        output = self.bge_model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        return output["dense_vecs"], output["lexical_weights"]

    def encode_e5(self, texts: List[str]) -> List:
        """Encode texts with E5 model."""
        # E5 requires "passage: " prefix for documents
        prefixed_texts = [f"passage: {t}" for t in texts]
        embeddings = self.e5_model.encode(prefixed_texts, convert_to_numpy=True)
        return embeddings

    def upload_batch(
        self,
        texts: List[str],
        cids: List[int],
        start_idx: int
    ):
        """Upload a batch of documents to all databases."""
        # Encode with BGE-m3
        bge_dense, bge_sparse = self.encode_bge(texts)

        # Encode with E5
        e5_embeddings = self.encode_e5(texts)

        # Prepare Qdrant points for BGE collection
        bge_points = []
        for i, (text, cid, dense, sparse) in enumerate(zip(texts, cids, bge_dense, bge_sparse)):
            point_id = start_idx + i

            # Convert sparse weights to Qdrant format
            sparse_indices = list(sparse.keys())
            sparse_values = list(sparse.values())

            bge_points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={"text": text, "cid": cid}
            ))

        # Upload to BGE collection
        self.qdrant.upsert(
            collection_name=BGE_COLLECTION,
            points=bge_points
        )

        # Prepare and upload to E5 collection
        e5_points = []
        for i, (text, cid, emb) in enumerate(zip(texts, cids, e5_embeddings)):
            point_id = start_idx + i
            e5_points.append(models.PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={"text": text, "cid": cid}
            ))

        self.qdrant.upsert(
            collection_name=E5_COLLECTION,
            points=e5_points
        )

        # Prepare and upload to Elasticsearch
        es_actions = []
        for i, (text, cid) in enumerate(zip(texts, cids)):
            es_actions.append({
                "_index": ELASTIC_INDEX,
                "_id": start_idx + i,
                "_source": {
                    "text": text,
                    "cid": cid
                }
            })

        helpers.bulk(self.es, es_actions)

    def save_checkpoint(self, idx: int):
        """Save checkpoint to file."""
        checkpoint = {"last_processed_idx": idx}
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f)
        logger.debug(f"Checkpoint saved at index {idx}")

    def load_checkpoint(self) -> int:
        """Load checkpoint from file."""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r") as f:
                checkpoint = json.load(f)
            return checkpoint.get("last_processed_idx", 0)
        return 0

    def import_data(
        self,
        corpus_path: str,
        percent: int = 10,
        resume: bool = False,
        recreate: bool = False
    ):
        """Import data from corpus.csv."""
        # Connect and setup
        self.connect()
        self.load_models()
        self.create_collections(recreate=recreate)

        # Load corpus
        logger.info(f"Loading corpus from {corpus_path}")
        df = pd.read_csv(corpus_path)
        total_rows = len(df)

        # Calculate rows to import based on percent
        rows_to_import = int(total_rows * percent / 100)
        df = df.head(rows_to_import)
        logger.info(f"Importing {percent}% of data: {rows_to_import:,} documents")

        # Get starting index
        start_idx = 0
        if resume:
            start_idx = self.load_checkpoint()
            logger.info(f"Resuming from index {start_idx}")

        # Process in batches
        texts = df["text"].tolist()
        cids = df["cid"].tolist()

        total_batches = (len(texts) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=len(texts) - start_idx, desc="Importing", unit="docs") as pbar:
            for batch_start in range(start_idx, len(texts), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(texts))

                batch_texts = texts[batch_start:batch_end]
                batch_cids = cids[batch_start:batch_end]

                try:
                    self.upload_batch(batch_texts, batch_cids, batch_start)
                except Exception as e:
                    logger.error(f"Error at batch {batch_start}: {e}")
                    self.save_checkpoint(batch_start)
                    raise

                pbar.update(batch_end - batch_start)

                # Save checkpoint periodically
                if batch_end % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(batch_end)

        # Final checkpoint
        self.save_checkpoint(len(texts))
        logger.info("Import completed successfully!")

        # Verify counts
        self.verify_import(rows_to_import)

    def verify_import(self, expected_count: int):
        """Verify import was successful."""
        logger.info("Verifying import...")

        bge_count = self.qdrant.get_collection(BGE_COLLECTION).points_count
        e5_count = self.qdrant.get_collection(E5_COLLECTION).points_count
        es_count = self.es.count(index=ELASTIC_INDEX)["count"]

        logger.info(f"BGE collection: {bge_count:,} points (expected: {expected_count:,})")
        logger.info(f"E5 collection: {e5_count:,} points (expected: {expected_count:,})")
        logger.info(f"Elasticsearch: {es_count:,} documents (expected: {expected_count:,})")

        if bge_count == e5_count == es_count == expected_count:
            logger.info("All counts match expected!")
        else:
            logger.warning("Counts do not match expected. Please verify.")


def main():
    parser = argparse.ArgumentParser(description="Import data into Qdrant and Elasticsearch")
    parser.add_argument("--corpus", required=True, help="Path to corpus.csv")
    parser.add_argument("--percent", type=int, default=10, help="Percentage of data to import (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--recreate", action="store_true", help="Recreate collections/index (delete existing data)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        logger.error(f"Corpus file not found: {args.corpus}")
        sys.exit(1)

    importer = DataImporter(use_gpu=not args.cpu)
    importer.import_data(
        corpus_path=args.corpus,
        percent=args.percent,
        resume=args.resume,
        recreate=args.recreate
    )


if __name__ == "__main__":
    main()
