import numpy as np
import pandas as pd
import torch


class MedDataset(torch.utils.data.Dataset):

    target_cols = ['heart_failure_1mo', 'heart_failure_3mo', 'heart_failure_6mo', 'heart_failure_12mo']
    binary_cols = ['male', 'has_heart_failure_outpatient']

    def __init__(self, file_path):
        """
        Custom dataset for loading and processing data for Transformer model.

        Args:
            file_path (str): Path to the Parquet file.
            binary_cols (list): List of binary column names.
        """
        self.df = pd.read_parquet(file_path)
        self.df = self.df.apply(pd.to_numeric, downcast='float')
        self.df.replace(-1, np.nan)
        self.ids = self.df['person_id'].unique()
        self.length = len(self.ids)

        drop_features = [col for col in self.df.columns if 'height' in col or 'weight' in col]
        features = self.df.drop(columns=self.target_cols + drop_features + ['person_id', 'time_block'])

        self.numeric_cols = [col for col in features.columns if col not in self.binary_cols]

        # Compute means and stds for numeric columns only
        features_numeric = features[self.numeric_cols]

        self.feature_means = {}
        self.feature_stds = {}
        for col in features_numeric.columns:
            self.feature_means[col] = features_numeric[col].mean()
            self.feature_stds[col] = features_numeric[col].std()
        # print(self.feature_means)
        # print(self.feature_stds)

        # Avoid division by zero in standard deviation
        self.feature_stds = {k: (v if v != 0 else 1) for k, v in self.feature_stds.items()}
        self.df.replace(np.nan, -np.inf)

    def normalize(self, x):
        """
        Apply Z-score normalization to the input data.
        """
        return (x - self.feature_means) / self.feature_stds

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
        - idx: Index of the sample.

        Returns:
        - x: Input matrix (features x time blocks) as a torch tensor.
        - y: Target values as a torch tensor.
        """
        # Get data for the specific person_id
        df_item = self.df[self.df['person_id'] == self.ids[idx]]
        df_item.reset_index()
        drop_features = [col for col in df_item.columns if 'height' in col or 'weight' in col]
        y = df_item[MedDataset.target_cols].iloc[0]
        x = df_item.drop(MedDataset.target_cols+drop_features+['person_id'], axis=1)
        x = x.sort_values(by='time_block', ascending=False)
        x = x.drop(['time_block'], axis=1)

        # Normalize features & Combine normalized numeric data and binary features
        # x_numeric = x[self.numeric_cols]
        # x_binary = x[self.binary_cols]
        # x_normalized = self.normalize(x_numeric)
        # x_combined = pd.concat([x_normalized, x_binary], axis=1)

        for f in self.numeric_cols:
            x[f] = (x[f] - self.feature_means[f])/self.feature_stds[f]

        # x_combined = x_combined.transpose()
        # x = x.transpose()
        y = y.transpose()
        # x_combined = x_combined.to_numpy(dtype=float)
        x = x.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)
        return x, y

    def __len__(self):
        return self.length


def main():
    file_path = "s3://datascience-dev-arcadia-io/disease-prediction/Cluster31/seq_to_seq_training_set1000.parquet"

    # Separate the target variables (outputs)
    # binary_cols = ['male', 'has_heart_failure_outpatient', 'heart_failure_1mo',
    #                'heart_failure_3mo', 'heart_failure_6mo', 'heart_failure_12mo']

    dataset = MedDataset(file_path)
    batch_size = 32
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the DataLoader and display shapes
    for i, (x, y) in enumerate(loader):
        print(f'x[{i}] shape: {x.shape}, y[{i}] shape: {y.shape}')


if __name__ == "__main__":
    main()

