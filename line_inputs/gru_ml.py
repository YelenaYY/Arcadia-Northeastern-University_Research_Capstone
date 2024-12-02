from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import gru_model
from matplotlib import pyplot as plt


class GruML():
    def __init__(self, hyper_params=None):

        self.DEVICE = torch.device('cpu')
        # self.DEVICE = torch.device('cuda')

        self.INPUT_SIZE = 79               # Number of features in your data
        self.OUTPUT_SIZE = 3               # Predicting heart failure over three time spans
        if hyper_params is None:
            self.GRU_HIDDEN_SIZE =64      # Number of GRU units (adjustable)
            self.GRU_NUM_LAYERS = 1        # Number of GRU layers (adjustable)
            self.GRU_DROPOUT = 0.3       # Dropout rate for regularization
            self.DENSE_DROPOUT = 0.3
            self.DENSE_HIDDEN_SIZE = 88
            self.BATCH_SIZE = 512
            self.LEARNING_RATE = 0.0001
            self.NUM_EPOCHS = 100
        else:
            self.GRU_HIDDEN_SIZE = hyper_params['GRU_HIDDEN_SIZE']
            self.DENSE_HIDDEN_SIZE = hyper_params['DENSE_HIDDEN_SIZE']
            self.GRU_NUM_LAYERS = hyper_params['GRU_NUM_LAYERS']
            self.GRU_DROPOUT = hyper_params['GRU_DROPOUT']
            self.DENSE_DROPOUT = hyper_params['DENSE_DROPOUT']
            self.BATCH_SIZE = hyper_params['BATCH_SIZE']
            self.LEARNING_RATE = hyper_params['LEARNING_RATE']
            self.NUM_EPOCHS = hyper_params['NUM_EPOCHS']

        self.model = gru_model.GruModel(
            self.INPUT_SIZE,
            self.GRU_HIDDEN_SIZE,
            self.GRU_NUM_LAYERS,
            self.GRU_DROPOUT,
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
                # print("inputshape")
                # print(inputs.shape)

                # Forward pass
                outputs = self.model(inputs)

                # print(outputs)
                # print(labels)
                
                # print("outputshape")
                # print(outputs.shape)
                # print(labels.shape)
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
        # plt.title("GRU Loss")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.savefig(f'gru-loss.png')
        # plt.clf()
        # plt.close()
            
        return loss_trace_test[-1]
