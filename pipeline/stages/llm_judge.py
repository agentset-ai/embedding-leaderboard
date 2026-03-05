"""
LLM Judge stage: AI-based model comparison
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
import requests
from sklearn.metrics.pairwise import cosine_similarity
import logging
from openai import OpenAI

from ..config import Config
from ..paths import RunPaths


def retrieve_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    k: int = 5
) -> List[str]:
    """Retrieve top-k documents for a query."""
    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [corpus_ids[idx] for idx in top_indices]


def call_azure_openai(
    prompt: str,
    api_key: str,
    resource_name: str,
    deployment_id: str,
    temperature: float = 0.0,
    max_tokens: int = 10
) -> str:
    """Call Azure OpenAI API (supports both v1 and legacy formats)."""
    # Check if we're using v1 format (resource_name contains full endpoint)
    if resource_name.startswith("http"):
        # New v1 format - resource_name is actually the full endpoint
        endpoint = resource_name
        client = OpenAI(base_url=endpoint, api_key=api_key)
        model = deployment_id

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert search quality evaluator. Respond with only: A, B, or TIE."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content.strip()
    else:
        # Legacy format - construct old-style endpoint
        endpoint = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version=2024-02-15-preview"

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert search quality evaluator. Respond with only: A, B, or TIE."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"].strip()


def create_judge_prompt(query: str, docs_a: List[str], docs_b: List[str], truncate_length: int = 200) -> str:
    """Create the AI judge prompt (blind comparison without model names)."""
    prompt = f"""Given a user query and two lists of search results from different embedding models, determine which model returns more relevant results. Return "A", "B" or "TIE".

The ordered list represents the relevance of the document to the query, the higher the position the more relevant.

Query: "{query}"

Ranking A (top {len(docs_a)}):
"""

    for i, doc in enumerate(docs_a, 1):
        doc_preview = doc[:truncate_length] + "..." if len(doc) > truncate_length else doc
        prompt += f"{i}. {doc_preview}\n"

    prompt += f"\nRanking B (top {len(docs_b)}):\n"

    for i, doc in enumerate(docs_b, 1):
        doc_preview = doc[:truncate_length] + "..." if len(doc) > truncate_length else doc
        prompt += f"{i}. {doc_preview}\n"

    prompt += "\nAnswer with exactly one token: A, B, or TIE."

    return prompt


def parse_judge_response(response: str) -> str:
    """Parse the judge's response."""
    response = response.strip().upper()

    if "A" in response and "B" not in response:
        return "A"
    elif "B" in response and "A" not in response:
        return "B"
    else:
        return "TIE"


def llm_judge_stage(config: Config, paths: RunPaths, logger: logging.Logger) -> Dict:
    """
    Run LLM judge comparison between embedders

    Args:
        config: Pipeline configuration
        paths: Run paths manager
        logger: Logger instance

    Returns:
        Dictionary with stage results metadata
    """
    if not config.llm_judge.enabled:
        logger.info("LLM judge disabled, skipping")
        return {'status': 'skipped', 'reason': 'disabled'}

    logger.info("Starting LLM judge stage...")

    if len(config.embedders) < 2:
        logger.info("Need at least 2 embedders for comparison, skipping")
        return {'status': 'skipped', 'reason': 'insufficient_models'}

    results = {
        'status': 'completed',
        'comparisons': []
    }

    for dataset in config.datasets:
        logger.info(f"LLM judge for dataset: {dataset.name}")
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

        # Compare all embedder pairs
        for i, embedder_a in enumerate(config.embedders):
            for embedder_b in config.embedders[i+1:]:
                logger.info(f"Comparing {embedder_a.name} vs {embedder_b.name}")
                print(f"   {embedder_a.name} vs {embedder_b.name}")

                # Reuse existing judge file if present
                judge_file = paths.get_llm_judge_file(dataset.name, embedder_a.name, embedder_b.name)
                if judge_file.exists() and config.pipeline.skip_if_exists:
                    print(f"      Already exists, reusing")
                    logger.info(f"Reusing existing judge file: {judge_file}")
                    with open(judge_file) as f:
                        existing = json.load(f)
                    results['comparisons'].append({
                        'dataset': dataset.name,
                        'model_a': embedder_a.name,
                        'model_b': embedder_b.name,
                        'wins_a': existing['wins_a'],
                        'wins_b': existing['wins_b']
                    })
                    continue

                # Load embeddings
                corpus_emb_a = np.load(paths.get_corpus_embedding_file(dataset.name, embedder_a.name))
                corpus_emb_b = np.load(paths.get_corpus_embedding_file(dataset.name, embedder_b.name))
                query_emb_a = np.load(paths.get_query_embedding_file(dataset.name, embedder_a.name))
                query_emb_b = np.load(paths.get_query_embedding_file(dataset.name, embedder_b.name))

                with open(paths.get_latency_file(dataset.name, embedder_a.name), 'r') as f:
                    latency_data_a = json.load(f)

                corpus_ids_a = latency_data_a['corpus_ids']
                corpus_ids_b = latency_data_a['corpus_ids']  # Same corpus
                query_ids = latency_data_a['query_ids']

                # Build ordered list of (id, text) pairs matching embedding order
                query_id_list = list(queries.keys())
                query_text_list = list(queries.values())

                # Sample queries
                random.seed(42)
                sampled_indices = random.sample(
                    range(len(query_ids)),
                    min(config.llm_judge.num_queries, len(query_ids))
                )

                wins_a = 0
                wins_b = 0
                ties = 0
                comparisons = []
                evaluated = 0

                for idx in sampled_indices:
                    query_id = query_ids[idx]
                    # query_ids may be positional indices or real IDs
                    if query_id in queries:
                        query_text = queries[query_id]
                    else:
                        # Fall back to positional lookup
                        query_text = query_text_list[idx]
                        query_id = query_id_list[idx]

                    # Retrieve documents
                    retrieved_ids_a = retrieve_top_k(
                        query_emb_a[idx], corpus_emb_a, corpus_ids_a, config.llm_judge.top_k
                    )
                    retrieved_ids_b = retrieve_top_k(
                        query_emb_b[idx], corpus_emb_b, corpus_ids_b, config.llm_judge.top_k
                    )

                    # Check if all retrieved docs exist in corpus (handle subset datasets)
                    missing_ids = []
                    for doc_id in retrieved_ids_a + retrieved_ids_b:
                        if doc_id not in corpus:
                            missing_ids.append(doc_id)

                    if missing_ids:
                        logger.warning(f"Skipping query {query_id}: {len(missing_ids)} retrieved docs not in corpus")
                        continue

                    retrieved_docs_a = [corpus[doc_id] for doc_id in retrieved_ids_a]
                    retrieved_docs_b = [corpus[doc_id] for doc_id in retrieved_ids_b]

                    # Randomly swap order to avoid position bias
                    swap = random.random() < 0.5
                    if swap:
                        docs_first = retrieved_docs_b
                        docs_second = retrieved_docs_a
                    else:
                        docs_first = retrieved_docs_a
                        docs_second = retrieved_docs_b

                    # Create prompt
                    prompt = create_judge_prompt(
                        query_text, docs_first, docs_second,
                        config.llm_judge.prompt_truncate_doc_length
                    )

                    # Call LLM judge
                    try:
                        response = call_azure_openai(
                            prompt,
                            config.llm_judge.azure_api_key,
                            config.llm_judge.azure_resource_name,
                            config.llm_judge.azure_deployment_id,
                            temperature=0.0
                        )

                        judgment = parse_judge_response(response)

                        # Map judgment back to actual models
                        if swap:
                            if judgment == "A":
                                actual_judgment = "B"
                            elif judgment == "B":
                                actual_judgment = "A"
                            else:
                                actual_judgment = "TIE"
                        else:
                            actual_judgment = judgment

                        if actual_judgment == "A":
                            wins_a += 1
                        elif actual_judgment == "B":
                            wins_b += 1
                        else:
                            ties += 1
                        evaluated += 1

                        comparisons.append({
                            'query_id': query_id,
                            'query': query_text,
                            'order_swapped': swap,
                            'llm_judgment': judgment,
                            'actual_judgment': actual_judgment,
                            'raw_response': response
                        })

                    except Exception as e:
                        logger.error(f"Error on query {idx + 1}: {str(e)}")
                        continue

                # Calculate Elo scores using iterative K-factor method
                total = wins_a + wins_b + ties
                if total == 0:
                    elo_a, elo_b = config.llm_judge.elo_initial_rating, config.llm_judge.elo_initial_rating
                else:
                    elo_a = float(config.llm_judge.elo_initial_rating)
                    elo_b = float(config.llm_judge.elo_initial_rating)
                    K = config.llm_judge.elo_k_factor
                    actual_a = (wins_a + 0.5 * ties) / total
                    actual_b = (wins_b + 0.5 * ties) / total
                    for _ in range(10):
                        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
                        expected_b = 1 - expected_a
                        elo_a += K * (actual_a - expected_a)
                        elo_b += K * (actual_b - expected_b)

                # Save results
                comparison_result = {
                    'dataset': dataset.name,
                    'model_a': embedder_a.name,
                    'model_b': embedder_b.name,
                    'num_queries': evaluated,
                    'wins_a': wins_a,
                    'wins_b': wins_b,
                    'ties': ties,
                    'win_rate_a': wins_a / total if total > 0 else 0,
                    'win_rate_b': wins_b / total if total > 0 else 0,
                    'elo_a': float(elo_a),
                    'elo_b': float(elo_b),
                    'comparisons': comparisons
                }

                judge_file = paths.get_llm_judge_file(dataset.name, embedder_a.name, embedder_b.name)
                with open(judge_file, 'w') as f:
                    json.dump(comparison_result, f, indent=2)

                print(f"      {embedder_a.name}: {wins_a} wins, {embedder_b.name}: {wins_b} wins, {ties} ties")
                logger.info(f"Saved comparison to {judge_file}")

                results['comparisons'].append({
                    'dataset': dataset.name,
                    'model_a': embedder_a.name,
                    'model_b': embedder_b.name,
                    'wins_a': wins_a,
                    'wins_b': wins_b
                })

    logger.info("LLM judge stage completed successfully")
    return results
