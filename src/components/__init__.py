# src/components/__init__.py
# Expose core pipeline components for easy import across the project.
#
# Usage:
#   from src.components import DataLoader, DataPreprocessor

from src.components.data_loader import DataLoader

__all__ = [
    "DataLoader",
]
