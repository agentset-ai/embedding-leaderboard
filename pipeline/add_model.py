#!/usr/bin/env python3
"""
Add a new model to the embedding leaderboard.

This script:
1. Generates embeddings for the new model across all datasets
2. Calculates NDCG/Recall metrics
3. Runs LLM judge comparisons against all existing models
4. Updates benchmarks.json with the new model's results

Usage:
    python -m pipeline.add_model --model zembed-1
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv


def load_model_info(results_dir: Path) -> Dict:
    """Load model-info.json"""
    with open(results_dir / "model-info.json") as f:
        models = json.load(f)
    return {m["name"]: m for m in models}


def load_benchmarks(results_dir: Path) -> List[Dict]:
    """Load current benchmarks.json"""
    benchmarks_file = results_dir / "benchmarks.json"
    if benchmarks_file.exists():
        with open(benchmarks_file) as f:
            return json.load(f)
    return []


def save_benchmarks(results_dir: Path, benchmarks: List[Dict]):
    """Save benchmarks.json"""
    with open(results_dir / "benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)


def get_existing_models(benchmarks: List[Dict]) -> List[str]:
    """Get list of model names already in benchmarks"""
    return [b["name"] for b in benchmarks]


def load_dataset(dataset_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load corpus, queries, and qrels from a dataset directory"""
    corpus = {}
    with open(dataset_path / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["text"]

    queries = {}
    with open(dataset_path / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels = {}
    with open(dataset_path / "qrels" / "test.tsv") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                query_id, _, doc_id, score = parts
            else:
                query_id, doc_id, score = parts
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(score)

    return corpus, queries, qrels


def get_embedding_client(model_name: str, model_info: Dict):
    """Create embedding client for the model"""
    from pipeline.stages.embed import get_client

    provider = model_info.get("provider", "").lower()

    # Map provider names to config provider names
    provider_map = {
        "openai": "openai",
        "voyage ai": "voyage",
        "cohere": "cohere",
        "jina ai": "jina",
        "google": "google",
        "qwen": "deepinfra",
        "baai": "deepinfra",
        "zeroentropy": "zeroentropy",
    }

    config_provider = provider_map.get(provider, provider)

    # Map model names to API model names
    model_map = {
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-004": "text-embedding-004",
        "voyage-3-large": "voyage-3-large",
        "voyage-3.5": "voyage-3.5",
        "voyage-3.5-lite": "voyage-3.5-lite",
        "voyage-4": "voyage-4",
        "cohere-embed-v3": "embed-english-v3.0",
        "cohere-embed-multilingual-v3": "embed-multilingual-v3.0",
        "jina-embeddings-v3": "jina-embeddings-v3",
        "jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small",
        "qwen3-embedding-8b": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "qwen3-embedding-4b": "Alibaba-NLP/gte-Qwen2-4B-instruct",
        "qwen3-embedding-0.6b": "Alibaba-NLP/gte-Qwen2-0.5B-instruct",
        "bge-m3": "BAAI/bge-m3",
        "zembed-1": "zembed-1",
    }

    api_model = model_map.get(model_name, model_name)

    # Get API key
    api_key_env_map = {
        "openai": "OPENAI_API_KEY",
        "voyage": "VOYAGE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "jina": "JINA_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepinfra": "DEEPINFRA_API_KEY",
        "zeroentropy": "ZEMBED_API_KEY",
    }

    api_key = os.getenv(api_key_env_map.get(config_provider, ""))
    if not api_key:
        raise ValueError(f"API key not set for {config_provider}")

    return get_client(config_provider, api_model, api_key)


def calculate_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    qrels: Dict,
    k_values: List[int] = [5, 10],
) -> Dict:
    """Calculate NDCG and Recall metrics"""
    from sklearn.metrics.pairwise import cosine_similarity

    # L2 normalize
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_embeddings = corpus_embeddings / np.maximum(corpus_norms, 1e-10)
    query_embeddings = query_embeddings / np.maximum(query_norms, 1e-10)

    results = {}
    for k in k_values:
        ndcg_scores = []
        recall_scores = []

        for i, query_id in enumerate(query_ids):
            if query_id not in qrels:
                continue

            # Get similarities and top-k
            sims = cosine_similarity([query_embeddings[i]], corpus_embeddings)[0]
            top_indices = np.argsort(sims)[::-1][:k]
            retrieved = [corpus_ids[idx] for idx in top_indices]

            # Relevance scores for NDCG
            relevance = [qrels[query_id].get(doc_id, 0) for doc_id in retrieved]

            # NDCG@k
            dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
            ideal = sorted(relevance, reverse=True)
            idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

            # Recall@k
            relevant = set(qrels[query_id].keys())
            retrieved_set = set(retrieved)
            recall = len(relevant & retrieved_set) / len(relevant) if relevant else 0
            recall_scores.append(recall)

        results[f"ndcg@{k}"] = float(np.mean(ndcg_scores)) if ndcg_scores else 0
        results[f"recall@{k}"] = float(np.mean(recall_scores)) if recall_scores else 0

    return results


def run_llm_judge(
    new_model: str,
    other_model: str,
    dataset_name: str,
    new_embeddings: Tuple[np.ndarray, np.ndarray],
    other_embeddings_path: Path,
    corpus: Dict,
    queries: Dict,
    corpus_ids: List[str],
    query_ids: List[str],
    llm_judge_dir: Path,
    num_queries: int = 10,
    top_k: int = 5,
    truncate_length: int = 200,
) -> Dict:
    """Run LLM judge comparison between two models"""
    from openai import OpenAI
    from sklearn.metrics.pairwise import cosine_similarity

    # Check if comparison already exists
    judge_file = llm_judge_dir / f"{dataset_name}_{new_model}_vs_{other_model}.json"
    reverse_file = llm_judge_dir / f"{dataset_name}_{other_model}_vs_{new_model}.json"

    if judge_file.exists():
        print(f"      Reusing existing: {judge_file.name}")
        with open(judge_file) as f:
            return json.load(f)
    if reverse_file.exists():
        print(f"      Reusing existing (reverse): {reverse_file.name}")
        with open(reverse_file) as f:
            data = json.load(f)
        # Swap wins
        return {
            "wins_a": data["wins_b"],
            "wins_b": data["wins_a"],
            "ties": data["ties"],
        }

    # Load other model's embeddings
    corpus_emb_other = np.load(other_embeddings_path / f"corpus_{other_model}.npy")
    query_emb_other = np.load(other_embeddings_path / f"queries_{other_model}.npy")

    corpus_emb_new, query_emb_new = new_embeddings

    # Get Azure OpenAI config
    api_key = os.getenv("AZURE_API_KEY")
    resource_name = os.getenv("AZURE_RESOURCE_NAME")
    deployment_id = os.getenv("AZURE_DEPLOYMENT_ID")

    if not all([api_key, resource_name, deployment_id]):
        print("      Skipping LLM judge (Azure credentials not set)")
        return {"wins_a": 0, "wins_b": 0, "ties": 0}

    client = OpenAI(base_url=resource_name, api_key=api_key)

    # Sample queries
    random.seed(42)
    sample_indices = random.sample(range(len(query_ids)), min(num_queries, len(query_ids)))

    wins_new = 0
    wins_other = 0
    ties = 0
    comparisons = []

    query_id_list = list(queries.keys())
    query_text_list = list(queries.values())

    for idx in sample_indices:
        query_id = query_ids[idx]
        if query_id in queries:
            query_text = queries[query_id]
        else:
            query_text = query_text_list[idx]
            query_id = query_id_list[idx]

        # Retrieve top-k for both models
        sims_new = cosine_similarity([query_emb_new[idx]], corpus_emb_new)[0]
        sims_other = cosine_similarity([query_emb_other[idx]], corpus_emb_other)[0]

        top_new = np.argsort(sims_new)[::-1][:top_k]
        top_other = np.argsort(sims_other)[::-1][:top_k]

        retrieved_new = [corpus_ids[i] for i in top_new]
        retrieved_other = [corpus_ids[i] for i in top_other]

        # Check all docs exist
        missing = [d for d in retrieved_new + retrieved_other if d not in corpus]
        if missing:
            continue

        docs_new = [corpus[d] for d in retrieved_new]
        docs_other = [corpus[d] for d in retrieved_other]

        # Random swap for position bias
        swap = random.random() < 0.5
        if swap:
            docs_a, docs_b = docs_other, docs_new
        else:
            docs_a, docs_b = docs_new, docs_other

        # Create prompt
        prompt = f'Given a user query and two lists of search results, determine which returns more relevant results. Return "A", "B" or "TIE".\n\nQuery: "{query_text}"\n\nRanking A:\n'
        for i, doc in enumerate(docs_a, 1):
            preview = doc[:truncate_length] + "..." if len(doc) > truncate_length else doc
            prompt += f"{i}. {preview}\n"
        prompt += "\nRanking B:\n"
        for i, doc in enumerate(docs_b, 1):
            preview = doc[:truncate_length] + "..." if len(doc) > truncate_length else doc
            prompt += f"{i}. {preview}\n"
        prompt += "\nAnswer with exactly one token: A, B, or TIE."

        try:
            completion = client.chat.completions.create(
                model=deployment_id,
                messages=[
                    {"role": "system", "content": "You are an expert search quality evaluator. Respond with only: A, B, or TIE."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            response = completion.choices[0].message.content.strip().upper()

            # Parse judgment
            if "A" in response and "B" not in response:
                judgment = "A"
            elif "B" in response and "A" not in response:
                judgment = "B"
            else:
                judgment = "TIE"

            # Map back if swapped
            if swap:
                actual = "B" if judgment == "A" else ("A" if judgment == "B" else "TIE")
            else:
                actual = judgment

            if actual == "A":
                wins_new += 1
            elif actual == "B":
                wins_other += 1
            else:
                ties += 1

            comparisons.append({
                "query_id": query_id,
                "llm_judgment": judgment,
                "actual_judgment": actual,
                "order_swapped": swap,
            })

        except Exception as e:
            print(f"      Error on query: {e}")
            continue

    # Save results
    result = {
        "dataset": dataset_name,
        "model_a": new_model,
        "model_b": other_model,
        "num_queries": len(comparisons),
        "wins_a": wins_new,
        "wins_b": wins_other,
        "ties": ties,
        "comparisons": comparisons,
    }

    with open(judge_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def calculate_elo(benchmarks: List[Dict], initial: int = 1500, k: int = 32) -> Dict[str, float]:
    """Calculate Elo scores from all pairwise comparisons"""
    # Initialize scores
    elo = {b["name"]: float(initial) for b in benchmarks}

    # Gather all comparisons
    comparisons = []
    for b in benchmarks:
        for other, data in b.get("comparisons", {}).items():
            if other in elo:
                comparisons.append({
                    "model_a": b["name"],
                    "model_b": other,
                    "wins_a": data.get("wins", 0),
                    "wins_b": data.get("losses", 0),
                    "ties": data.get("ties", 0),
                })

    # Run 10 iterative passes
    for _ in range(10):
        for comp in comparisons:
            model_a, model_b = comp["model_a"], comp["model_b"]
            total = comp["wins_a"] + comp["wins_b"] + comp["ties"]
            if total == 0:
                continue

            actual_a = (comp["wins_a"] + 0.5 * comp["ties"]) / total
            actual_b = (comp["wins_b"] + 0.5 * comp["ties"]) / total

            expected_a = 1 / (1 + 10 ** ((elo[model_b] - elo[model_a]) / 400))
            expected_b = 1 - expected_a

            elo[model_a] += k * (actual_a - expected_a)
            elo[model_b] += k * (actual_b - expected_b)

    return elo


def main():
    parser = argparse.ArgumentParser(description="Add a new model to the leaderboard")
    parser.add_argument("--model", required=True, help="Model name (must match model-info.json)")
    parser.add_argument("--datasets-dir", default="datasets", help="Path to datasets directory")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    args = parser.parse_args()

    load_dotenv()

    datasets_dir = Path(args.datasets_dir)
    results_dir = Path(args.results_dir)
    llm_judge_dir = results_dir / "llm_judge"
    llm_judge_dir.mkdir(exist_ok=True)

    # Load model info
    model_info_all = load_model_info(results_dir)
    if args.model not in model_info_all:
        print(f"Error: Model '{args.model}' not found in model-info.json")
        sys.exit(1)

    model_info = model_info_all[args.model]
    print(f"\nAdding model: {args.model}")
    print(f"  Provider: {model_info['provider']}")
    print(f"  Dimension: {model_info['dimension']}")

    # Load current benchmarks
    benchmarks = load_benchmarks(results_dir)
    existing_models = get_existing_models(benchmarks)

    if args.model in existing_models:
        print(f"\nWarning: Model '{args.model}' already in benchmarks. Will update.")
        benchmarks = [b for b in benchmarks if b["name"] != args.model]
        existing_models.remove(args.model)

    # Get embedding client
    try:
        client = get_embedding_client(args.model, model_info)
    except Exception as e:
        print(f"Error creating embedding client: {e}")
        sys.exit(1)

    # Process each dataset
    datasets = ["fiqa", "msmarco", "scifact", "dbpedia", "business-reports", "arcd", "pg"]

    new_benchmark = {
        "name": model_info["display_name"],
        "overall": {
            "elo": 1500.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "win_rate": 0.0,
            "total_judgments": 0,
            "avg_latency_ms": 0.0,
            "avg_ndcg_10": 0.0,
        },
        "by_dataset": {},
        "comparisons": {},
    }

    all_latencies = []
    all_ndcg_10 = []

    for dataset_name in datasets:
        dataset_path = datasets_dir / dataset_name
        if not dataset_path.exists():
            print(f"\nSkipping {dataset_name}: not found")
            continue

        print(f"\nProcessing {dataset_name}...")

        # Load dataset
        corpus, queries, qrels = load_dataset(dataset_path)
        corpus_ids = list(corpus.keys())
        corpus_texts = list(corpus.values())
        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        print(f"  Corpus: {len(corpus)} docs, Queries: {len(queries)}")

        # Generate embeddings
        print(f"  Embedding corpus...")
        corpus_emb = client.embed_corpus(corpus_texts)
        print(f"  Embedding queries...")
        query_emb, latencies = client.embed_queries(query_texts)

        all_latencies.extend(latencies)

        # Calculate metrics
        print(f"  Calculating metrics...")
        metrics = calculate_metrics(query_emb, corpus_emb, query_ids, corpus_ids, qrels)
        all_ndcg_10.append(metrics["ndcg@10"])

        # Store dataset results
        new_benchmark["by_dataset"][dataset_name.upper().replace("-", "_")] = {
            "elo": 1500.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "win_rate": 0.0,
            "metrics": metrics,
            "latency": {
                "mean_ms": float(np.mean(latencies)) * 1000,
                "p50_ms": float(np.percentile(latencies, 50)) * 1000,
                "p90_ms": float(np.percentile(latencies, 90)) * 1000,
            },
        }

        print(f"  NDCG@10: {metrics['ndcg@10']:.4f}, Recall@10: {metrics['recall@10']:.4f}")

        # Run LLM judge against existing models
        if existing_models:
            print(f"  Running LLM judge comparisons...")
            for other_model in existing_models:
                # Find embeddings path for other model
                emb_path = None
                for possible_dir in [
                    Path("data/embeddings"),
                    Path("embeddings"),
                    results_dir.parent / "embedder-leaderboard" / "embeddings" / "20260206_120931",
                ]:
                    test_path = possible_dir / dataset_name
                    if test_path.exists() and (test_path / f"corpus_{other_model}.npy").exists():
                        emb_path = test_path
                        break

                if not emb_path:
                    print(f"    Skipping {other_model}: embeddings not found")
                    continue

                print(f"    vs {other_model}...", end=" ")
                result = run_llm_judge(
                    args.model,
                    other_model,
                    dataset_name,
                    (corpus_emb, query_emb),
                    emb_path,
                    corpus,
                    queries,
                    corpus_ids,
                    query_ids,
                    llm_judge_dir,
                )
                print(f"W:{result['wins_a']} L:{result['wins_b']} T:{result['ties']}")

                # Aggregate comparisons
                if other_model not in new_benchmark["comparisons"]:
                    new_benchmark["comparisons"][other_model] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
                new_benchmark["comparisons"][other_model]["wins"] += result["wins_a"]
                new_benchmark["comparisons"][other_model]["losses"] += result["wins_b"]
                new_benchmark["comparisons"][other_model]["ties"] += result["ties"]
                new_benchmark["comparisons"][other_model]["total"] += result["wins_a"] + result["wins_b"] + result["ties"]

                # Update overall stats
                new_benchmark["overall"]["wins"] += result["wins_a"]
                new_benchmark["overall"]["losses"] += result["wins_b"]
                new_benchmark["overall"]["ties"] += result["ties"]
                new_benchmark["overall"]["total_judgments"] += result["wins_a"] + result["wins_b"] + result["ties"]

    # Calculate overall stats
    total = new_benchmark["overall"]["wins"] + new_benchmark["overall"]["losses"] + new_benchmark["overall"]["ties"]
    if total > 0:
        new_benchmark["overall"]["win_rate"] = new_benchmark["overall"]["wins"] / total

    new_benchmark["overall"]["avg_latency_ms"] = float(np.mean(all_latencies)) * 1000 if all_latencies else 0
    new_benchmark["overall"]["avg_ndcg_10"] = float(np.mean(all_ndcg_10)) if all_ndcg_10 else 0

    # Add to benchmarks
    benchmarks.append(new_benchmark)

    # Recalculate Elo scores for all models
    print("\nRecalculating Elo scores...")
    elo_scores = calculate_elo(benchmarks)

    for benchmark in benchmarks:
        model_name = benchmark["name"]
        if model_name in elo_scores:
            benchmark["overall"]["elo"] = round(elo_scores[model_name], 2)

    # Sort by Elo
    benchmarks.sort(key=lambda x: x["overall"]["elo"], reverse=True)

    # Save updated benchmarks
    save_benchmarks(results_dir, benchmarks)
    print(f"\nUpdated benchmarks.json with {len(benchmarks)} models")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Elo: {new_benchmark['overall']['elo']:.0f}")
    print(f"Win Rate: {new_benchmark['overall']['win_rate']:.2%}")
    print(f"Avg NDCG@10: {new_benchmark['overall']['avg_ndcg_10']:.4f}")
    print(f"Avg Latency: {new_benchmark['overall']['avg_latency_ms']:.2f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
