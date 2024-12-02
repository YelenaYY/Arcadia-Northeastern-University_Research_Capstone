# run_transformer.py

import torch
import torch.nn as nn
import pandas as pd
# import torch.optim as optim
from transformer_ml import TransformerML
from data_preprocessor import DataPreprocessor, Tokenizer
from transformer_model import TransformerModel
from data_loader import get_data_loaders

torch.manual_seed(0)

# # Configuration parameters
# embedding_dim = 64
# hidden_size = 128
# nheads = 4
# n_layers = 2
# dropout = 0.1
# max_src_len = 500  # Based on expected source sequence length
# max_tgt_len = 3    # Since we are predicting 3 target intervals

# File path and column definitions
S3_PATH = "Place Holder"
file_path = S3_PATH
binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']

# Model parameters
# model_params = {
#     "input_vocab_size": train_loader.dataset[0][0].shape[0],
#     "output_vocab_size": 2,  # Assuming binary classification
#     "embedding_dim": 64,
#     "hidden_size": 128,
#     "num_heads": 4,
#     "num_layers": 2,
#     "max_src_len": train_loader.dataset[0][0].shape[0],  # Set to input feature size
#     "max_tgt_len": len(target_cols),  # Number of target columns
#     "dropout": 0.1
# }

# training_params = {
#     "num_epochs": 10,
#     "batch_size": 32,
#     "learning_rate": 0.001
# }

# Step 4: Initialize TransformerML with Parameters
transformer_ml = TransformerML(
    # device="cpu",
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # model_params=model_params,
    # training_params=training_params
)

# Get train and test DataLoaders
train_loader, test_loader = get_data_loaders(file_path, binary_cols, target_cols, batch_size=transformer_ml.BATCH_SIZE, split_ratio=0.8)


# Step 5: Fit the Model
transformer_ml.fit(train_loader, test_loader)