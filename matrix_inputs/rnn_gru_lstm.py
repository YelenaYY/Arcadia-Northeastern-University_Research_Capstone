import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from med_dataset import MedDataset
from data_loader import get_data_loaders
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_sequence_len,
                 input_size_rnn,
                 hidden_size_rnn,
                 num_layers_rnn,
                 dropout_rnn,
                 output_sequence_len,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 device):
        super(RNN, self).__init__()

        # Hyper-parameters
        self.rnn_type = rnn_type
        self.input_sequence_len = input_sequence_len
        self.input_size_rnn = input_size_rnn
        self.hidden_size_rnn = hidden_size_rnn
        self.num_layers_rnn = num_layers_rnn
        self.dropout_rnn = dropout_rnn
        self.output_sequence_len = output_sequence_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Define RNN type (LSTM/GRU)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size_rnn,
                               hidden_size_rnn,
                               num_layers_rnn,
                               dropout=dropout_rnn,
                               bidirectional=False,
                               batch_first=True,
                               dtype=torch.float64)
        else:
            self.rnn = nn.GRU(input_size_rnn,
                              hidden_size_rnn,
                              num_layers_rnn,
                              dropout=dropout_rnn,
                              bidirectional=False,
                              batch_first=True,
                              dtype=torch.float64)

        # Fully connected Layer
        self.dense = nn.Linear(hidden_size_rnn * input_sequence_len, output_sequence_len, dtype=torch.float64)

        # Criterion/optimizer
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Hardware device
        self.device = torch.device(device)

    def forward(self, src):
        h0 = torch.zeros(self.num_layers_rnn, src.size(0), self.hidden_size_rnn,
                         dtype=torch.float64).to(device=self.device)
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers_rnn, src.size(0), self.hidden_size_rnn,
                             dtype=torch.float64).to(device=self.device)
            rnn_out, _ = self.rnn(src, (h0, c0))
        else:
            rnn_out, _ = self.rnn(src, h0)
        rnn_out = rnn_out.reshape(rnn_out.size(0), -1)
        out = self.dense(rnn_out)
        return out


class RnnModel:
    def __init__(self, hyperparams):
        DEVICE = 'cpu'
        self.device = torch.device(DEVICE)
        self.model = RNN(hyperparams['RNN_TYPE'],
                         hyperparams['INPUT_SEQUENCE_LEN'],
                         hyperparams['NUM_INPUT_SEQUENCES'],
                         hyperparams['RNN_HIDDEN_SIZE'],
                         hyperparams['RNN_NUM_LAYERS'],
                         hyperparams['RNN_DROPOUT'],
                         hyperparams['OUTPUT_SEQUENCE_LEN'],
                         hyperparams['LEARNING_RATE'],
                         hyperparams['BATCH_SIZE'],
                         hyperparams['NUM_EPOCHS'],
                         DEVICE).to(self.device)

    def fit(self, train_loader, test_loader):

        # train_dataset = MedDataset(train_file_path)
        # test_dataset = MedDataset(test_file_path)

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.model.batch_size, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.model.batch_size, shuffle=True)
        loss_trace_train = []
        loss_trace_test = []

        for epoch in tqdm(range(self.model.num_epochs)):

            # Train
            current_loss_train = 0
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x = torch.permute(x, [0, 2, 1])

                # Forward pass
                outputs = self.model(x)
                outputs = torch.squeeze(outputs)

                # Compute loss
                loss = self.model.criterion(outputs, y)

                # Reset gradients
                self.model.optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                self.model.optimizer.step()
                current_loss_train += loss.item()
            current_loss_train /= i
            loss_trace_train.append(current_loss_train)

            # Test
            current_loss_test = 0
            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    x = torch.permute(x, [0, 2, 1])
                    outputs = self.model(x)
                    outputs = torch.squeeze(outputs)
                    loss = self.model.criterion(outputs, y)
                    current_loss_test += loss.item()
            self.model.train()
            current_loss_test /= i
            loss_trace_test.append(current_loss_test)

        # loss curve
        plt.plot(range(1, self.model.num_epochs + 1), [n for n in loss_trace_train], 'r-', label='Train Loss')
        plt.plot(range(1, self.model.num_epochs + 1), [m for m in loss_trace_test], 'b-', label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig(f'loss_curve_rnn.png')
        # plt.clf()
        # plt.close()


def main():
    file_path = "S3_Path Placeholder"
    train_loader, test_loader = get_data_loaders(file_path)

    hyperparams = {'RNN_TYPE': 'GRU',
         'INPUT_SEQUENCE_LEN': 12,
        'NUM_INPUT_SEQUENCES': 29,
            'RNN_HIDDEN_SIZE': 2,
             'RNN_NUM_LAYERS': 2,
                'RNN_DROPOUT': 0,
        'OUTPUT_SEQUENCE_LEN': 4,
              'LEARNING_RATE': 0.003,
                 'BATCH_SIZE': 10,
                 'NUM_EPOCHS': 100}
    
    gru_model = RnnModel(hyperparams)
    gru_model.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()