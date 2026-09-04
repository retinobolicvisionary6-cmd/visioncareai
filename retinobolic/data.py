"""
Data Handling Module for Retinobolic.
Handles dataset loading, cleaning, patient-level splits, and PyTorch DataLoaders.
"""
from src.dataset import (
    FundusDataset,
    inspect_and_clean_dataset,
    create_splits,
    get_dataloaders,
    map_raw_label_to_4class
)

__all__ = [
    "FundusDataset",
    "inspect_and_clean_dataset",
    "create_splits",
    "get_dataloaders",
    "map_raw_label_to_4class"
]
