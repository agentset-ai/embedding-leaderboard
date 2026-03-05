"""
Configuration loader for embedder evaluation pipeline
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    base_path: str
    corpus_file: str
    queries_file: str
    qrels_file: Optional[str] = None

    @property
    def corpus_path(self) -> Path:
        return Path(self.base_path) / self.corpus_file

    @property
    def queries_path(self) -> Path:
        return Path(self.base_path) / self.queries_file

    @property
    def qrels_path(self) -> Optional[Path]:
        if self.qrels_file is None:
            return None
        return Path(self.base_path) / self.qrels_file


@dataclass
class EmbedderConfig:
    """Embedder model configuration"""
    name: str
    provider: str  # voyage, cohere, openai, deepinfra, jina, google, zeroentropy
    model: str
    api_key_env: str

    @property
    def api_key(self) -> Optional[str]:
        if self.api_key_env is None or self.api_key_env == "":
            return None
        return os.getenv(self.api_key_env)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    k_values: List[int] = field(default_factory=lambda: [5, 10])
    metrics: List[str] = field(default_factory=lambda: ["ndcg", "recall"])
    generate_plots: bool = True


@dataclass
class LLMJudgeConfig:
    """LLM Judge configuration"""
    enabled: bool = True
    provider: str = "azure_openai"
    azure_api_key_env: str = "AZURE_API_KEY"
    azure_resource_name_env: str = "AZURE_RESOURCE_NAME"
    azure_deployment_id_env: str = "AZURE_DEPLOYMENT_ID"
    num_queries: int = 10
    top_k: int = 5
    elo_initial_rating: int = 1500
    elo_k_factor: int = 32
    prompt_truncate_doc_length: int = 200

    @property
    def azure_api_key(self) -> Optional[str]:
        return os.getenv(self.azure_api_key_env)

    @property
    def azure_resource_name(self) -> Optional[str]:
        return os.getenv(self.azure_resource_name_env)

    @property
    def azure_deployment_id(self) -> Optional[str]:
        return os.getenv(self.azure_deployment_id_env)


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    stages: List[str] = field(default_factory=lambda: ["embed", "evaluate", "llm_judge", "visualize"])
    skip_if_exists: bool = False


@dataclass
class Config:
    """Main configuration container"""
    datasets: List[DatasetConfig]
    embedders: List[EmbedderConfig]
    evaluation: EvaluationConfig
    llm_judge: LLMJudgeConfig
    pipeline: PipelineConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Flatten nested llm_judge config (elo and prompt sections)
        llm_judge_data = data.get('llm_judge', {}).copy()
        if 'elo' in llm_judge_data:
            llm_judge_data['elo_initial_rating'] = llm_judge_data['elo'].get('initial_rating', 1500)
            llm_judge_data['elo_k_factor'] = llm_judge_data['elo'].get('k_factor', 32)
            del llm_judge_data['elo']
        if 'prompt' in llm_judge_data:
            llm_judge_data['prompt_truncate_doc_length'] = llm_judge_data['prompt'].get('truncate_doc_length', 200)
            del llm_judge_data['prompt']

        return cls(
            datasets=[DatasetConfig(**d) for d in data['datasets']],
            embedders=[EmbedderConfig(**e) for e in data['embedders']],
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            llm_judge=LLMJudgeConfig(**llm_judge_data),
            pipeline=PipelineConfig(**data.get('pipeline', {}))
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate dataset paths exist
        for dataset in self.datasets:
            if not dataset.corpus_path.exists():
                errors.append(f"Corpus file not found: {dataset.corpus_path}")
            if not dataset.queries_path.exists():
                errors.append(f"Queries file not found: {dataset.queries_path}")
            if dataset.qrels_path is not None and not dataset.qrels_path.exists():
                errors.append(f"Qrels file not found: {dataset.qrels_path}")

        # Validate embedders have API keys
        for embedder in self.embedders:
            if embedder.api_key_env and embedder.api_key is None:
                errors.append(f"API key not set for embedder {embedder.name} (env: {embedder.api_key_env})")

        # Validate LLM judge if enabled
        if self.llm_judge.enabled:
            if self.llm_judge.provider == "azure_openai":
                if not self.llm_judge.azure_api_key:
                    errors.append(f"Azure API key not set (env: {self.llm_judge.azure_api_key_env})")
                if not self.llm_judge.azure_resource_name:
                    errors.append(f"Azure resource name not set (env: {self.llm_judge.azure_resource_name_env})")
                if not self.llm_judge.azure_deployment_id:
                    errors.append(f"Azure deployment ID not set (env: {self.llm_judge.azure_deployment_id_env})")

        return errors
