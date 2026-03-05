"""
Evaluation stage: Calculate NDCG and Recall metrics
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..config import Config
from ..paths import RunPaths


def calculate_ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Calculate NDCG@k (Normalized Discounted Cumulative Gain)."""
    relevance_scores = relevance_scores[:k]

    if not relevance_scores:
        return 0.0

    # DCG: sum of (relevance / log2(position + 1))
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    # IDCG: DCG of ideal ranking (sorted by relevance)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_recall_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
    """Calculate Recall@k."""
    if not relevant_docs:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    num_relevant_retrieved = len(retrieved_at_k & relevant_set)
    return num_relevant_retrieved / len(relevant_set)


def retrieve_and_rank(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    top_k: int = 100
) -> List[str]:
    """Retrieve and rank documents for a query."""
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return document IDs
    return [corpus_ids[idx] for idx in top_indices]


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: List[str],
    corpus_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [5, 10]
) -> Dict:
    """Evaluate retrieval performance."""
    results = {
        'num_queries': len(query_ids),
        'k_values': k_values,
        'metrics': {}
    }

    # Initialize metric accumulators
    for k in k_values:
        results['metrics'][f'ndcg@{k}'] = []
        results['metrics'][f'recall@{k}'] = []

    # Evaluate each query
    for i, query_id in enumerate(query_ids):
        query_embedding = query_embeddings[i]

        # Get relevant documents
        if query_id not in qrels or not qrels[query_id]:
            continue

        relevant_docs = list(qrels[query_id].keys())

        # Retrieve and rank documents
        max_k = max(k_values)
        retrieved_docs = retrieve_and_rank(
            query_embedding,
            corpus_embeddings,
            corpus_ids,
            top_k=max_k
        )

        # Calculate metrics for each k
        for k in k_values:
            # Get relevance scores for retrieved docs
            relevance_scores = [
                qrels[query_id].get(doc_id, 0)
                for doc_id in retrieved_docs[:k]
            ]

            # Calculate NDCG@k
            ndcg = calculate_ndcg_at_k(relevance_scores, k)
            results['metrics'][f'ndcg@{k}'].append(ndcg)

            # Calculate Recall@k
            recall = calculate_recall_at_k(relevant_docs, retrieved_docs, k)
            results['metrics'][f'recall@{k}'].append(recall)

    # Calculate average metrics
    for metric_name, scores in results['metrics'].items():
        results[metric_name] = float(np.mean(scores)) if scores else 0.0

    return results


def calculate_elo_scores(results_list: List[Dict], initial_elo: int = 1500, k: int = 32) -> Dict[str, float]:
    """Calculate Elo scores from pairwise comparisons."""
    # Initialize Elo scores
    elo_scores = {result['model']: initial_elo for result in results_list}

    # Perform pairwise comparisons
    for i, result_a in enumerate(results_list):
        for result_b in results_list[i + 1:]:
            model_a = result_a['model']
            model_b = result_b['model']

            # Compare on NDCG@10 (primary metric)
            score_a = result_a.get('ndcg@10', 0)
            score_b = result_b.get('ndcg@10', 0)

            # Calculate expected scores
            expected_a = 1 / (1 + 10 ** ((elo_scores[model_b] - elo_scores[model_a]) / 400))
            expected_b = 1 / (1 + 10 ** ((elo_scores[model_a] - elo_scores[model_b]) / 400))

            # Determine actual scores (1 for win, 0.5 for tie, 0 for loss)
            if score_a > score_b:
                actual_a, actual_b = 1.0, 0.0
            elif score_a < score_b:
                actual_a, actual_b = 0.0, 1.0
            else:
                actual_a, actual_b = 0.5, 0.5

            # Update Elo scores
            elo_scores[model_a] += k * (actual_a - expected_a)
            elo_scores[model_b] += k * (actual_b - expected_b)

    return elo_scores


def evaluate_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Evaluate retrieval performance for all datasets and embedders

    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance

    Returns:
        Dictionary with stage results metadata
    """
    logger.info("Starting evaluation stage...")

    results = {
        'status': 'completed',
        'datasets': {}
    }

    for dataset in config.datasets:
        logger.info(f"Evaluating dataset: {dataset.name}")
        print(f"\nDataset: {dataset.name}")

        # Load qrels
        qrels = {}
        with open(dataset.qrels_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                # Handle both 3-column and 4-column formats
                if len(parts) == 4:
                    query_id, _, doc_id, score = parts
                else:
                    query_id, doc_id, score = parts
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(score)

        logger.info(f"Loaded {len(qrels)} qrels")

        dataset_results = []

        for embedder in config.embedders:
            logger.info(f"Evaluating {embedder.name}")
            print(f"   Model: {embedder.name}")

            # Load embeddings
            corpus_file = paths.get_corpus_embedding_file(dataset.name, embedder.name)
            query_file = paths.get_query_embedding_file(dataset.name, embedder.name)
            latency_file = paths.get_latency_file(dataset.name, embedder.name)

            if not corpus_file.exists() or not query_file.exists():
                logger.warning(f"Embeddings not found for {embedder.name}, skipping")
                print(f"      Embeddings not found, skipping")
                continue

            corpus_embeddings = np.load(corpus_file)
            query_embeddings = np.load(query_file)

            # L2-normalize all embeddings for consistent cosine similarity
            corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            corpus_embeddings = corpus_embeddings / np.maximum(corpus_norms, 1e-10)
            query_embeddings = query_embeddings / np.maximum(query_norms, 1e-10)

            with open(latency_file, 'r') as f:
                latency_data = json.load(f)

            corpus_ids = latency_data['corpus_ids']
            query_ids = latency_data['query_ids']

            # Evaluate retrieval
            print(f"      Evaluating retrieval...")
            eval_results = evaluate_retrieval(
                query_embeddings,
                corpus_embeddings,
                query_ids,
                corpus_ids,
                qrels,
                config.evaluation.k_values
            )

            # Add model info and latency
            eval_results['model'] = embedder.name
            eval_results['dataset'] = dataset.name
            eval_results['avg_query_latency'] = latency_data['avg_latency']
            eval_results['min_query_latency'] = latency_data['min_latency']
            eval_results['max_query_latency'] = latency_data['max_latency']

            # Save results
            results_file = paths.get_evaluation_file(dataset.name, embedder.name)
            with open(results_file, 'w') as f:
                json.dump(eval_results, f, indent=2)

            print(f"      Results saved")
            for k in config.evaluation.k_values:
                logger.info(f"NDCG@{k}: {eval_results[f'ndcg@{k}']:.4f}, Recall@{k}: {eval_results[f'recall@{k}']:.4f}")

            dataset_results.append(eval_results)

        # Compare models for this dataset
        if len(dataset_results) > 1:
            logger.info(f"Comparing models on {dataset.name}")
            print(f"\n   Comparing {len(dataset_results)} models...")

            elo_scores = calculate_elo_scores(dataset_results)

            comparison = {
                'dataset': dataset.name,
                'num_models': len(dataset_results),
                'elo_scores': elo_scores,
                'results': dataset_results
            }

            comparison_file = paths.get_comparison_file(dataset.name)
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2)

            print(f"   Comparison saved")
            logger.info(f"Comparison saved to {comparison_file}")

        results['datasets'][dataset.name] = {
            'num_models': len(dataset_results),
            'models': [r['model'] for r in dataset_results]
        }

    # Generate global leaderboard
    logger.info("Generating global leaderboard...")
    print(f"\nGenerating global leaderboard...")

    model_scores = {}
    for dataset in config.datasets:
        for embedder in config.embedders:
            eval_file = paths.get_evaluation_file(dataset.name, embedder.name)
            if not eval_file.exists():
                continue

            with open(eval_file, 'r') as f:
                result = json.load(f)

            model = result['model']
            if model not in model_scores:
                model_scores[model] = {
                    'ndcg@5': [],
                    'ndcg@10': [],
                    'recall@5': [],
                    'recall@10': [],
                    'latency': [],
                    'datasets': []
                }

            for k in config.evaluation.k_values:
                model_scores[model][f'ndcg@{k}'].append(result.get(f'ndcg@{k}', 0))
                model_scores[model][f'recall@{k}'].append(result.get(f'recall@{k}', 0))
            model_scores[model]['latency'].append(result.get('avg_query_latency', 0))
            model_scores[model]['datasets'].append(dataset.name)

    # Calculate averages
    global_leaderboard = []
    for model, scores in model_scores.items():
        entry = {
            'model': model,
            'num_datasets': len(scores['datasets']),
            'datasets': scores['datasets']
        }
        for k in config.evaluation.k_values:
            entry[f'avg_ndcg@{k}'] = float(np.mean(scores[f'ndcg@{k}']))
            entry[f'avg_recall@{k}'] = float(np.mean(scores[f'recall@{k}']))
        entry['avg_latency'] = float(np.mean(scores['latency']))
        global_leaderboard.append(entry)

    # Sort by NDCG@10
    global_leaderboard.sort(key=lambda x: x.get('avg_ndcg@10', 0), reverse=True)

    # Save global leaderboard
    leaderboard_file = paths.get_global_leaderboard_file()
    with open(leaderboard_file, 'w') as f:
        json.dump(global_leaderboard, f, indent=2)

    print(f"Global leaderboard saved")
    logger.info(f"Global leaderboard saved to {leaderboard_file}")

    results['global_leaderboard'] = global_leaderboard

    logger.info("Evaluation stage completed successfully")
    return results
