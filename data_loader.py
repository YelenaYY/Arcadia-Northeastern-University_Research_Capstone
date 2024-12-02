import torch
import pandas as pd
import numpy as np
from med_dataset import MedDataset
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset


def get_data_loaders(file_path, batch_size=32, split_ratio=0.8):
    """
    Prepare data loaders for training and testing sets.
    
    Args:
        file_path (str): Path to the Parquet file.
        binary_cols (list): List of binary column names.
        target_cols (list): List of target column names.
        batch_size (int): Batch size for DataLoader.
        split_ratio (float): Proportion of data for training.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders.
    """
    # Initialize dataset
    dataset = MedDataset(file_path)

    # Split dataset into training and testing sets
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

