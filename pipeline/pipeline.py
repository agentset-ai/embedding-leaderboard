"""
Main pipeline orchestrator
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .config import Config
from .paths import RunPaths
from .logger import PipelineLogger
from .stages import (
    embed_stage,
    evaluate_stage,
    llm_judge_stage,
    visualize_stage
)


class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: str = "config.yaml", timestamp: str = None):
        """
        Initialize pipeline with configuration

        Args:
            config_path: Path to YAML configuration file
            timestamp: Optional timestamp to reuse existing run directory
        """
        self.config = Config.from_yaml(config_path)
        self.errors = self.config.validate()

        if self.errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in self.errors))

        # Initialize run paths
        self.paths = RunPaths(timestamp=timestamp)

        # Store results
        self.results = {}

    def run(self, stages: list = None):
        """
        Run the pipeline

        Args:
            stages: Optional list of stages to run (defaults to config.pipeline.stages)
        """
        if stages is None:
            stages = self.config.pipeline.stages

        print("\n" + "="*80)
        print("EMBEDDER EVALUATION PIPELINE")
        print("="*80)
        print(f"Datasets: {', '.join([d.name for d in self.config.datasets])}")
        print(f"Embedders: {', '.join([e.name for e in self.config.embedders])}")
        print(f"Run directory: {self.paths.run_dir}")
        print(f"Stages to run: {', '.join(stages)}")
        print(f"Total stages: {len(stages)}")
        print("="*80)

        # Save metadata
        metadata = {
            'timestamp': self.paths.timestamp,
            'run_dir': str(self.paths.run_dir),
            'config': {
                'datasets': [d.name for d in self.config.datasets],
                'embedders': [{'name': e.name, 'provider': e.provider, 'model': e.model}
                             for e in self.config.embedders],
                'k_values': self.config.evaluation.k_values,
                'metrics': self.config.evaluation.metrics,
                'llm_judge_enabled': self.config.llm_judge.enabled
            }
        }
        self.paths.save_metadata(metadata)

        # Run stages
        stage_map = {
            'embed': embed_stage,
            'evaluate': evaluate_stage,
            'llm_judge': llm_judge_stage,
            'visualize': visualize_stage
        }

        total_stages = len(stages)
        for stage_idx, stage_name in enumerate(stages, 1):
            if stage_name not in stage_map:
                print(f"\nWarning: Unknown stage '{stage_name}', skipping...")
                continue

            print(f"\n{'='*80}")
            print(f"STAGE {stage_idx}/{total_stages}: {stage_name.upper()}")
            print(f"{'='*80}")

            with PipelineLogger(self.paths, stage_name) as logger:
                try:
                    stage_func = stage_map[stage_name]
                    print(f"Running {stage_name}...")
                    result = stage_func(self.config, self.paths, logger)
                    self.results[stage_name] = result
                    status = result.get('status', 'unknown')

                    if status == 'completed':
                        print(f"Stage '{stage_name}' completed successfully!")
                    elif status == 'skipped':
                        print(f"Stage '{stage_name}' skipped (already exists)")
                    else:
                        print(f"Stage '{stage_name}' finished: {status}")

                except Exception as e:
                    print(f"\nERROR: Stage '{stage_name}' failed!")
                    print(f"   Error: {str(e)}")
                    logger.error(f"Stage '{stage_name}' failed: {e}", exc_info=True)
                    self.results[stage_name] = {'status': 'failed', 'error': str(e)}
                    raise

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved to: {self.paths.run_dir}")
        print("\nOutput Summary:")
        for dataset in self.config.datasets:
            print(f"   - {dataset.name}: {self.paths.get_dataset_dir(dataset.name)}")
        if self.config.llm_judge.enabled:
            print(f"   - LLM Judge: {self.paths.llm_judge_dir}")
        print(f"   - Visualizations: {self.paths.visualizations_dir}")
        print(f"   - Report: {self.paths.get_report_file()}")
        print("="*80 + "\n")

        return self.results


def main():
    """Main entry point"""
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    timestamp = sys.argv[2] if len(sys.argv) > 2 else None

    pipeline = Pipeline(config_path, timestamp=timestamp)
    pipeline.run()


if __name__ == '__main__':
    main()
