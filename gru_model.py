import torch
import torch.nn as nn
import torch.optim as optim


class GruModel(nn.Module):
    def __init__(self, input_size, gru_hidden_size, gru_num_layers, gru_dropout, 
                 dense_hidden_size, dense_dropout, output_size, learning_rate):
        super(GruModel, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=gru_hidden_size,
                          num_layers=gru_num_layers,
                          dropout=gru_dropout,
                          batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(gru_hidden_size, dense_hidden_size)
        self.fc2 = nn.Linear(dense_hidden_size, output_size)

        # Sigmoid layer
        self.out = nn.Sigmoid()

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dense_dropout)

        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for multi-label classification
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.DEVICE = torch.device('cpu')
        # self.DEVICE = torch.device('cuda')

        self.to(self.DEVICE)

    def forward(self, x):
        # print('FORWARD START')
        # print(x.shape)
        x = torch.unsqueeze(x, 1)
        # print(x.shape)
        # GRU layer
        _, h_n = self.gru(x)  # We only need the last hidden state for classification
        # print(h_n.shape)
        # Reshape hidden state and pass through fully connected layers
        out = self.fc1(h_n[-1])
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.dropout(out)
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        out = self.out(out)
        # print('FORWARD END')

        return out
