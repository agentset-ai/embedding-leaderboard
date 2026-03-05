"""
Embedding stage: Generate embeddings for corpus and queries
"""

import json
import time
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
import logging

from ..config import Config
from ..paths import RunPaths


# ============================================================================
# EMBEDDING CLIENTS
# ============================================================================

class VoyageClient:
    """Voyage AI embedding client."""

    def __init__(self, api_key: str, model_name: str = "voyage-3-large"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.voyageai.com/v1/embeddings"

    def embed_corpus(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            batch = [text if text and text.strip() else " " for text in batch]
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": batch,
                    "model": self.model_name,
                    "input_type": "document"
                }
            )
            if response.status_code != 200:
                print(f"\nAPI Error (batch {i//batch_size + 1}):")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
            response.raise_for_status()
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": [query],
                    "model": self.model_name,
                    "input_type": "query"
                }
            )
            latency = time.time() - start_time
            response.raise_for_status()

            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class OpenAIClient:
    """OpenAI embedding client."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-large"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/embeddings"

    def embed_corpus(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            batch = [text if text and text.strip() else " " for text in batch]
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": batch,
                    "model": self.model_name
                }
            )
            if response.status_code != 200:
                print(f"\nAPI Error: {response.status_code} {response.text}")
            response.raise_for_status()
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": [query],
                    "model": self.model_name
                }
            )
            latency = time.time() - start_time
            response.raise_for_status()

            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class CohereClient:
    """Cohere embedding client."""

    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.cohere.ai/v1/embed"

    def embed_corpus(self, texts: List[str], batch_size: int = 96) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "texts": batch,
                    "model": self.model_name,
                    "input_type": "search_document"
                }
            )
            response.raise_for_status()
            batch_embeddings = response.json()["embeddings"]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "texts": [query],
                    "model": self.model_name,
                    "input_type": "search_query"
                }
            )
            latency = time.time() - start_time
            response.raise_for_status()

            embedding = response.json()["embeddings"][0]
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class JinaClient:
    """Jina AI embedding client."""

    def __init__(self, api_key: str, model_name: str = "jina-embeddings-v3"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def _post_with_retry(self, payload: dict, max_retries: int = 5) -> dict:
        """POST to Jina API with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"\n      Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if response.status_code != 200:
                print(f"\nAPI Error: {response.status_code} {response.text}")
            response.raise_for_status()
            return response.json()
        raise Exception("Max retries exceeded on Jina rate limit")

    def embed_corpus(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            batch = [text if text and text.strip() else " " for text in batch]
            data = self._post_with_retry({
                "input": batch,
                "model": self.model_name,
                "task": "retrieval.passage",
                "normalized": True
            })
            batch_embeddings = [item["embedding"] for item in data["data"]]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            data = self._post_with_retry({
                "input": [query],
                "model": self.model_name,
                "task": "retrieval.query",
                "normalized": True
            })
            latency = time.time() - start_time

            embedding = data["data"][0]["embedding"]
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class GoogleClient:
    """Google Gemini embedding client."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:embedContent"

    def _embed_single(self, text: str, task_type: str) -> List[float]:
        """Embed a single text."""
        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "model": f"models/{self.model_name}",
                "content": {"parts": [{"text": text}]},
                "taskType": task_type
            }
        )
        if response.status_code != 200:
            print(f"\nAPI Error: {response.status_code} {response.text}")
        response.raise_for_status()
        return response.json()["embedding"]["values"]

    def embed_corpus(self, texts: List[str], batch_size: int = 1) -> np.ndarray:
        """Embed corpus texts (one at a time for Google API)."""
        embeddings = []
        for text in tqdm(texts, desc=f"Embedding corpus ({self.model_name})"):
            text = text if text and text.strip() else " "
            embedding = self._embed_single(text, "RETRIEVAL_DOCUMENT")
            embeddings.append(embedding)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            embedding = self._embed_single(query, "RETRIEVAL_QUERY")
            latency = time.time() - start_time
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class DeepInfraClient:
    """DeepInfra embedding client for Qwen3 and BGE models."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.deepinfra.com/v1/openai/embeddings"

    def embed_corpus(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            batch = [text if text and text.strip() else " " for text in batch]
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": batch,
                    "model": self.model_name
                }
            )
            if response.status_code != 200:
                print(f"\nAPI Error: {response.status_code} {response.text}")
            response.raise_for_status()
            batch_embeddings = [item["embedding"] for item in response.json()["data"]]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": [query],
                    "model": self.model_name
                }
            )
            latency = time.time() - start_time
            response.raise_for_status()

            embedding = response.json()["data"][0]["embedding"]
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


class ZeroEntropyClient:
    """ZeroEntropy zembed embedding client."""

    def __init__(self, api_key: str, model_name: str = "zembed-1"):
        self.api_key = api_key
        self.model_name = model_name
        # EU endpoint as specified
        self.api_url = "https://eu-api.zeroentropy.dev/v1/models/embed"

    def embed_corpus(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed corpus texts with progress bar."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding corpus ({self.model_name})"):
            batch = texts[i:i + batch_size]
            batch = [text if text and text.strip() else " " for text in batch]
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": batch,
                    "model": self.model_name,
                    "input_type": "document"
                }
            )
            if response.status_code != 200:
                print(f"\nAPI Error: {response.status_code} {response.text}")
            response.raise_for_status()
            data = response.json()
            # Handle ZeroEntropy response format: {"results": [{"embedding": [...]}], "usage": {...}}
            if "results" in data:
                batch_embeddings = [item["embedding"] for item in data["results"]]
            elif "embeddings" in data:
                batch_embeddings = data["embeddings"]
            elif "data" in data:
                batch_embeddings = [item["embedding"] for item in data["data"]]
            else:
                raise ValueError(f"Unexpected response format: {data.keys()}")
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        """Embed queries and return embeddings + latencies."""
        embeddings = []
        latencies = []

        for query in tqdm(queries, desc=f"Embedding queries ({self.model_name})"):
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": [query],
                    "model": self.model_name,
                    "input_type": "query"
                }
            )
            latency = time.time() - start_time
            response.raise_for_status()

            data = response.json()
            # Handle ZeroEntropy response format
            if "results" in data:
                embedding = data["results"][0]["embedding"]
            elif "embeddings" in data:
                embedding = data["embeddings"][0]
            elif "data" in data:
                embedding = data["data"][0]["embedding"]
            else:
                raise ValueError(f"Unexpected response format: {data.keys()}")
            embeddings.append(embedding)
            latencies.append(latency)

        return np.array(embeddings), latencies


def get_client(provider: str, model_name: str, api_key: str):
    """Factory function to get embedding client."""
    if provider == "voyage":
        return VoyageClient(api_key, model_name)
    elif provider == "openai":
        return OpenAIClient(api_key, model_name)
    elif provider == "cohere":
        return CohereClient(api_key, model_name)
    elif provider == "jina":
        return JinaClient(api_key, model_name)
    elif provider == "google":
        return GoogleClient(api_key, model_name)
    elif provider == "deepinfra":
        return DeepInfraClient(api_key, model_name)
    elif provider == "zeroentropy":
        return ZeroEntropyClient(api_key, model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ============================================================================
# EMBED STAGE
# ============================================================================

def embed_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Generate embeddings for all datasets and embedders

    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance

    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting embedding stage...")

    results = {
        'status': 'completed',
        'datasets': {},
        'total_embedders': len(config.embedders),
        'total_datasets': len(config.datasets)
    }

    for dataset in config.datasets:
        logger.info(f"Processing dataset: {dataset.name}")
        print(f"\nDataset: {dataset.name}")

        # Load corpus
        corpus = {}
        with open(dataset.corpus_path, 'r') as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc['_id']] = doc['text']

        # Load queries
        queries = {}
        with open(dataset.queries_path, 'r') as f:
            for line in f:
                query = json.loads(line)
                queries[query['_id']] = query['text']

        corpus_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())
        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        logger.info(f"Loaded {len(corpus)} docs, {len(queries)} queries")

        dataset_results = {}

        for embedder in config.embedders:
            logger.info(f"Embedding with {embedder.name}")
            print(f"   Model: {embedder.name}")

            # Check if embeddings already exist
            corpus_file = paths.get_corpus_embedding_file(dataset.name, embedder.name)
            query_file = paths.get_query_embedding_file(dataset.name, embedder.name)
            latency_file = paths.get_latency_file(dataset.name, embedder.name)

            if (corpus_file.exists() and query_file.exists() and
                latency_file.exists() and config.pipeline.skip_if_exists):
                logger.info(f"Embeddings already exist, skipping...")
                print(f"      Skipping (already exists)")
                continue

            # Create client
            client = get_client(embedder.provider, embedder.model, embedder.api_key)

            # Embed corpus
            print(f"      Embedding corpus...")
            corpus_start = time.time()
            corpus_embeddings = client.embed_corpus(corpus_texts)
            corpus_time = time.time() - corpus_start
            logger.info(f"Corpus embedded in {corpus_time:.2f}s, shape: {corpus_embeddings.shape}")

            # Embed queries
            print(f"      Embedding queries...")
            query_embeddings, query_latencies = client.embed_queries(query_texts)
            logger.info(f"Queries embedded, shape: {query_embeddings.shape}")

            # Save embeddings
            np.save(corpus_file, corpus_embeddings)
            np.save(query_file, query_embeddings)

            # Save latency data
            latency_data = {
                'query_latencies': query_latencies,
                'avg_latency': float(np.mean(query_latencies)),
                'min_latency': float(np.min(query_latencies)),
                'max_latency': float(np.max(query_latencies)),
                'corpus_time': corpus_time,
                'query_ids': query_ids,
                'corpus_ids': corpus_ids
            }

            with open(latency_file, 'w') as f:
                json.dump(latency_data, f, indent=2)

            print(f"      Saved embeddings")
            logger.info(f"Saved embeddings to {corpus_file}")

            dataset_results[embedder.name] = {
                'corpus_shape': corpus_embeddings.shape,
                'query_shape': query_embeddings.shape,
                'avg_latency': latency_data['avg_latency']
            }

        results['datasets'][dataset.name] = dataset_results

    logger.info("Embedding stage completed successfully")
    return results
