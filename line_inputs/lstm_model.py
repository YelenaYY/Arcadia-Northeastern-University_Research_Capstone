import torch
import torch.nn as nn
import torch.optim as optim

class LstmModel(nn.Module):
    
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout, 
                 dense_hidden_size, dense_dropout, output_size, learning_rate):
        super(LstmModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            dropout=lstm_dropout,
                            batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, dense_hidden_size)
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
        x = torch.unsqueeze(x, 1)

        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.DEVICE)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.DEVICE)

        # LSTM layer
        _, (h_n, _) = self.lstm(x, (h0, c0))  # Output only the hidden state from the last time step

        # Reshape hidden state and pass through fully connected layers
        out = self.fc1(h_n[-1])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.out(out)
        # print('FORWARD END')

        return out
