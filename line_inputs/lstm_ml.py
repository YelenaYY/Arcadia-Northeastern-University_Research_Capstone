from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import lstm_model
from matplotlib import pyplot as plt

class LstmML():
    def __init__(self, hyper_params=None):

        self.DEVICE = torch.device('cpu')
        # self.DEVICE = torch.device('cuda')

        self.INPUT_SIZE = 79               # Number of features in your data
        self.OUTPUT_SIZE = 3               # Predicting heart failure over three time spans
        if hyper_params is None:
            self.LSTM_HIDDEN_SIZE = 16      # Number of LSTM units (adjustable)
            self.LSTM_NUM_LAYERS = 2        # Number of LSTM layers (adjustable)
            self.LSTM_DROPOUT = 0.2         # Dropout rate for regularization
            self.DENSE_DROPOUT = 0.2
            self.DENSE_HIDDEN_SIZE = 16
            self.BATCH_SIZE = 100
            self.LEARNING_RATE = 0.01
            self.NUM_EPOCHS = 20
        else:
            self.LSTM_HIDDEN_SIZE = hyper_params['LSTM_HIDDEN_SIZE']
            self.DENSE_HIDDEN_SIZE = hyper_params['DENSE_HIDDEN_SIZE']
            self.LSTM_NUM_LAYERS = hyper_params['LSTM_NUM_LAYERS']
            self.LSTM_DROPOUT = hyper_params['LSTM_DROPOUT']
            self.DENSE_DROPOUT = hyper_params['DENSE_DROPOUT']
            self.BATCH_SIZE = hyper_params['BATCH_SIZE']
            self.LEARNING_RATE = hyper_params['LEARNING_RATE']
            self.NUM_EPOCHS = hyper_params['NUM_EPOCHS']

        # Use the LstmModel instead of GruModel
        self.model = lstm_model.LstmModel(
            self.INPUT_SIZE,
            self.LSTM_HIDDEN_SIZE,
            self.LSTM_NUM_LAYERS,
            self.LSTM_DROPOUT,
            self.DENSE_HIDDEN_SIZE,
            self.DENSE_DROPOUT,
            self.OUTPUT_SIZE,
            self.LEARNING_RATE).to(self.DEVICE)

        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for multi-label classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def fit(self, train_data, test_data):

        loss_trace = []
        loss_trace_test = []

        for epoch in tqdm(range(self.NUM_EPOCHS)):
            # Set model to training mode
            self.model.train()
            current_loss_train = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_data):
                # Move data to the target device
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                # Forward pass
                outputs = self.model(inputs)
                
                labels = labels.float()
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                current_loss_train += loss.item()

            # Calculate average training loss
            avg_train_loss = current_loss_train / len(train_data)

            # Validation phase
            self.model.eval()
            current_loss_test = 0.0
            with torch.no_grad():
                for inputs, labels in test_data:
                    inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                    # Forward pass
                    outputs = self.model(inputs)
                    labels = labels.float()
                    loss = self.criterion(outputs, labels)
                    current_loss_test += loss.item()

            # Calculate average validation loss
            avg_test_loss = current_loss_test / len(test_data)

            loss_trace.append(current_loss_train)
            loss_trace_test.append(current_loss_test)

        # loss curve
        # plt.plot(range(1, self.NUM_EPOCHS + 1), [n/max(loss_trace) for n in loss_trace], 'r-')
        # plt.plot(range(1, self.NUM_EPOCHS + 1), [n/max(loss_trace_test) for n in loss_trace_test], 'b-')
        # plt.title("LSTM Loss")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.savefig(f'lstm-loss.png')
        # plt.clf()
        # plt.close()
            
        return loss_trace_test[-1]
