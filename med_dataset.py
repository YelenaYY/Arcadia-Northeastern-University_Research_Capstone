import numpy as np
import pandas as pd
import torch


class MedDataset(torch.utils.data.Dataset):

    target_cols = ['heart_failure_1mo', 'heart_failure_3mo', 'heart_failure_6mo', 'heart_failure_12mo']

    def __init__(self, file_path):
        """
        Custom dataset for loading and processing data for Transformer model.

        Args:
            file_path (str): Path to the Parquet file.
            binary_cols (list): List of binary column names.
        """
        self.df = pd.read_parquet(file_path)
        self.ids = self.df['person_id'].unique()
        self.length = len(self.ids)

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
        x = x.transpose()
        y = y.transpose()
        x = x.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)
        return x, y

    def __len__(self):
        return self.length


def main():
    file_path = "S3_Path Placeholder"

    # Separate the target variables (outputs)
    binary_cols = ['male', 'has_heart_failure_outpatient', 'heart_failure_1mo',
                   'heart_failure_3mo', 'heart_failure_6mo', 'heart_failure_12mo']

    dataset = MedDataset(file_path)
    batch_size = 32
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the DataLoader and display shapes
    for i, (x, y) in enumerate(loader):
        print(f'x[{i}] shape: {x.shape}, y[{i}] shape: {y.shape}')


if __name__ == "__main__":
    main()
