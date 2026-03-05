"""
Logging utilities for pipeline
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Context manager for pipeline stage logging"""

    def __init__(self, paths, stage_name: str):
        """
        Initialize logger for a pipeline stage

        Args:
            paths: RunPaths instance
            stage_name: Name of the stage
        """
        self.paths = paths
        self.stage_name = stage_name
        self.log_file = paths.run_dir / f"{stage_name}.log"

        # Create logger
        self.logger = logging.getLogger(f"pipeline.{stage_name}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def __enter__(self):
        """Enter context"""
        self.logger.info(f"Starting stage: {self.stage_name}")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        if exc_type is not None:
            self.logger.error(f"Stage {self.stage_name} failed with error: {exc_val}")
        else:
            self.logger.info(f"Stage {self.stage_name} completed")

        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

        return False  # Don't suppress exceptions
