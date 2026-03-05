"""
Path management for pipeline runs
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class RunPaths:
    """Manages paths for a pipeline run"""

    def __init__(self, base_runs_dir: str = "data/embeddings", timestamp: Optional[str] = None):
        """
        Initialize run paths

        Args:
            base_runs_dir: Base directory for runs
            timestamp: Optional timestamp string (defaults to current time)
        """
        self.base_runs_dir = Path(base_runs_dir)

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        self.run_dir = self.base_runs_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories per dataset and model
        self.embeddings_dir = self.run_dir / "embeddings"
        self.evaluation_dir = self.run_dir / "evaluation"
        self.llm_judge_dir = self.run_dir / "llm_judge"
        self.visualizations_dir = self.run_dir / "visualizations"

        for dir_path in [self.embeddings_dir, self.evaluation_dir,
                         self.llm_judge_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: dict):
        """Save metadata JSON to run directory"""
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Get directory for a specific dataset"""
        dataset_dir = self.run_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def get_corpus_embedding_file(self, dataset_name: str, embedder_name: str) -> Path:
        """Get path for corpus embedding file"""
        return self.get_dataset_dir(dataset_name) / f"corpus_{embedder_name}.npy"

    def get_query_embedding_file(self, dataset_name: str, embedder_name: str) -> Path:
        """Get path for query embedding file"""
        return self.get_dataset_dir(dataset_name) / f"queries_{embedder_name}.npy"

    def get_latency_file(self, dataset_name: str, embedder_name: str) -> Path:
        """Get path for latency JSON"""
        return self.get_dataset_dir(dataset_name) / f"latencies_{embedder_name}.json"

    def get_evaluation_file(self, dataset_name: str, embedder_name: str) -> Path:
        """Get path for evaluation results"""
        return self.get_dataset_dir(dataset_name) / f"eval_{embedder_name}.json"

    def get_comparison_file(self, dataset_name: str) -> Path:
        """Get path for model comparison file"""
        return self.get_dataset_dir(dataset_name) / "comparison.json"

    def get_llm_judge_file(self, dataset_name: str, model_a: str, model_b: str) -> Path:
        """Get path for LLM judge comparison"""
        return self.llm_judge_dir / f"{dataset_name}_{model_a}_vs_{model_b}.json"

    def get_global_leaderboard_file(self) -> Path:
        """Get path for global leaderboard"""
        return self.run_dir / "global_leaderboard.json"

    def get_report_file(self) -> Path:
        """Get path for HTML report"""
        return self.run_dir / "report.html"

    def __repr__(self):
        return f"RunPaths(timestamp={self.timestamp}, run_dir={self.run_dir})"
