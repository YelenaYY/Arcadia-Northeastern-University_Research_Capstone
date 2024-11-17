import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    def __init__(self, binary_cols, numerical_cols, categorical_cols=[]):
        self.binary_cols = binary_cols
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df):
        # Process numerical columns
        numerical_data = self.scaler.fit_transform(df[self.numerical_cols])
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)

        # Process binary columns
        binary_data = df[self.binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        binary_tensor = torch.tensor(binary_data, dtype=torch.float32)

        # Process categorical columns if any
        if self.categorical_cols:
            categorical_data = []
            for col in self.categorical_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.label_encoders[col] = encoder
                categorical_data.append(df[col].values)

            categorical_data = torch.tensor(categorical_data, dtype=torch.long).T
            return torch.cat((numerical_tensor, binary_tensor, categorical_data), dim=1)

        return torch.cat((numerical_tensor, binary_tensor), dim=1)

    def transform(self, df):
        # Similar to fit_transform but using fitted scalers and encoders
        numerical_data = self.scaler.transform(df[self.numerical_cols])
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)

        binary_data = df[self.binary_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        binary_tensor = torch.tensor(binary_data, dtype=torch.float32)

        if self.categorical_cols:
            categorical_data = []
            for col in self.categorical_cols:
                encoder = self.label_encoders[col]
                categorical_data.append(encoder.transform(df[col].values))

            categorical_data = torch.tensor(categorical_data, dtype=torch.long).T
            return torch.cat((numerical_tensor, binary_tensor, categorical_data), dim=1)

        return torch.cat((numerical_tensor, binary_tensor), dim=1)


class Tokenizer:
    def __init__(self, binary_cols, numerical_cols, seq_intervals):
        self.binary_cols = binary_cols
        self.numerical_cols = numerical_cols
        self.seq_intervals = seq_intervals  # Expecting ['1mo', '12mo', '120mo']

    def tokenize(self, df):
        # Initialize dictionaries to store the sequences
        tokenized_data = {}

        for interval in self.seq_intervals:
            interval_cols = [col for col in self.numerical_cols if f"_{interval}" in col]
            # print(interval_cols)
            interval_data = df[interval_cols]

            # Standardize the numerical data within each interval (or apply your existing scaler)
            interval_tensor = torch.tensor(interval_data.values, dtype=torch.float32)

            # Collect binary columns related to heart failure prediction
            heart_failure_cols = [col for col in self.binary_cols if f"heart_failure_{interval}" in col]
            heart_failure_data = df[heart_failure_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            heart_failure_tensor = torch.tensor(heart_failure_data.values, dtype=torch.float32)

            # Concatenate numerical and binary tensors along the feature dimension
            tokenized_data[interval] = torch.cat((interval_tensor, heart_failure_tensor), dim=1)

        return tokenized_data

    def get_target_sequences(self, df):
        # Prepare target sequences for each interval
        targets = {}
        for interval in self.seq_intervals:
            target_col = f"heart_failure_{interval}"
            target_data = df[target_col].apply(pd.to_numeric, errors='coerce').fillna(0)
            targets[interval] = torch.tensor(target_data.values, dtype=torch.float32).unsqueeze(1)
        return targets

