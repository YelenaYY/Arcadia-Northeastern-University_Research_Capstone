# run_gru.py

import torch
import torch.nn as nn
import pandas as pd
from data_preprocessor import DataPreprocessor
import gru_ml
from data_loader import get_data_loaders

torch.manual_seed(0)

# File path and column definitions
S3_PATH = "Place Holder"
file_path = S3_PATH
binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']


# Initialize TransformerML with Parameters
gru = gru_ml.GruML()

# Get train and test DataLoaders
train_loader, test_loader = get_data_loaders(file_path,
                                             binary_cols,
                                             target_cols,
                                             batch_size=gru.BATCH_SIZE,
                                             split_ratio=0.8)


# Step 5: Fit the Model
gru.fit(train_loader, test_loader)
