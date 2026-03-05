# Adding a New Model to the Leaderboard

## Step 1: Add to model-info.json

Add your model's metadata to `results/model-info.json`:

```json
{
  "name": "your-model-name",
  "display_name": "Your Model Display Name",
  "provider": "Provider Name",
  "license": "License Type",
  "cost_per_1m_tokens": 0.05,
  "release_date": "2026-01-01",
  "dimension": 1024,
  "about_model": "Brief description of the model."
}
```

**Important**: The `name` field is the join key used everywhere (filenames, config, benchmarks).

## Step 2: Add Embedder Client (if new provider)

If your model uses a new API provider, add a client class to `pipeline/stages/embed.py`:

```python
class YourProviderClient:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.yourprovider.com/v1/embeddings"

    def embed_corpus(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # Implement corpus embedding
        pass

    def embed_queries(self, queries: List[str]) -> Tuple[np.ndarray, List[float]]:
        # Implement query embedding with latency tracking
        pass
```

Then add to the `get_client()` factory function:

```python
elif provider == "yourprovider":
    return YourProviderClient(api_key, model_name)
```

## Step 3: Add to config.yaml

Add your model to the `embedders` section:

```yaml
embedders:
  # ... existing models ...

  - name: "your-model-name"
    provider: "yourprovider"
    model: "api-model-id"
    api_key_env: "YOUR_API_KEY"
```

## Step 4: Run Evaluation

```bash
# Set your API key
export YOUR_API_KEY="sk-..."
export AZURE_API_KEY="..."  # For LLM judge

# Run the add_model script
python -m pipeline.add_model --model your-model-name

# Or run the full pipeline
python -m pipeline config.yaml
```

## Step 5: Verify Results

Check that your model appears in:

- `results/benchmarks.json` - Should have Elo scores and metrics
- `results/llm_judge/` - Should have comparison files vs all other models

## Key Configuration Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_queries` | 10 | Queries per model pair per dataset for LLM judge |
| `top_k` | 5 | Documents compared in judge prompt |
| `truncate_doc_length` | 200 | Max chars per document in prompt |
| `elo_initial_rating` | 1500 | Starting Elo rating |
| `elo_k_factor` | 32 | Elo adjustment factor |
| `max_tokens` | 10 | LLM judge response limit (prevents verbose output) |

## Notes

- All embeddings are L2-normalized before evaluation
- LLM judge uses position-swapping to reduce bias
- Existing judge files are reused (never regenerated) when `skip_if_exists: true`
