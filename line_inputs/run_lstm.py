# run_lstm.py

import torch
import torch.nn as nn
import pandas as pd
from data_preprocessor import DataPreprocessor
import lstm_ml  # Import the LSTM-based model
from data_loader import get_data_loaders

torch.manual_seed(0)

# File path and column definitions
S3_PATH = "Place Holder"
file_path = S3_PATH
binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']

# Initialize LstmML with Parameters
lstm = lstm_ml.LstmML()  # Replace GruML with LstmML

# Get train and test DataLoaders
train_loader, test_loader = get_data_loaders(file_path,
                                             binary_cols,
                                             target_cols,
                                             batch_size=lstm.BATCH_SIZE,
                                             split_ratio=0.8)

# Fit the Model
lstm.fit(train_loader, test_loader)  # Use lstm instance to fit
