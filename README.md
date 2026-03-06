# Embedding Model Leaderboard

A benchmark for comparing embedding models using Elo ratings from LLM-as-judge pairwise comparisons.

View current results in `results/benchmarks.json`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
cp .env.example .env  # Edit with your keys

# Run the full pipeline
python -m pipeline config.yaml
```

## Adding a New Model

See [ADD_MODEL.md](ADD_MODEL.md) for instructions.

## Project Structure

```
embedding-leaderboard-final/
├── config.yaml          # Pipeline configuration
├── pipeline/            # Pipeline code
│   ├── stages/          # embed, evaluate, llm_judge, visualize
│   └── add_model.py     # Script to add new models
├── datasets/            # 7 evaluation datasets
├── results/             # Benchmark results and LLM judge files
│   ├── model-info.json  # Model metadata
│   ├── benchmarks.json  # Current leaderboard
│   └── llm_judge/       # Pairwise comparison results
└── data/embeddings/     # Generated embeddings (gitignored)
```
