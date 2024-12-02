import pandas as pd
import numpy as np
# import os
# import time
import transformer_model
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

torch.manual_seed(0)


class TransformerML():

    def __init__(self, hyper_params=None):
        self.INPUT_VOCAB_SIZE = 1
        self.OUTPUT_VOCAB_SIZE = 2  # 0 or 1
        self.MAX_SRC_LEN = 79
        self.INPUT_SIZE = 79
        self.MAX_TGT_LEN = 3

        if hyper_params is None:
            self.NUM_EPOCHS = 22
            self.HIDDEN_SIZE = 3
            self.EMBEDDING_DIM = 8   # Must be divisible by NUM_HEADS
            self.BATCH_SIZE = 800
            self.NUM_HEADS = 8        # Must divide EMBEDDING_DIM
            self.NUM_LAYERS = 4
            self.LEARNING_RATE = 0.01
            self.DROPOUT = 0.2
        else:
            self.NUM_EPOCHS = hyper_params['NUM_EPOCHS']
            self.HIDDEN_SIZE = hyper_params['HIDDEN_SIZE']
            self.EMBEDDING_DIM = hyper_params['EMBEDDING_DIM_FACTOR'] * hyper_params['NUM_HEADS']
            self.BATCH_SIZE = hyper_params['BATCH_SIZE']
            self.NUM_HEADS = hyper_params['NUM_HEADS']
            self.NUM_LAYERS = hyper_params['NUM_LAYERS']
            self.LEARNING_RATE = hyper_params['LEARNING_RATE']
            self.DROPOUT = hyper_params['DROPOUT']

        self.DEVICE = torch.device('cpu')
        # self.DEVICE = torch.device('cuda')
        self.model = transformer_model.TransformerModel(
            self.INPUT_SIZE,
            self.INPUT_VOCAB_SIZE,
            self.OUTPUT_VOCAB_SIZE,
            self.EMBEDDING_DIM,
            self.HIDDEN_SIZE,
            self.NUM_HEADS,
            self.NUM_LAYERS,
            self.MAX_SRC_LEN,
            self.MAX_TGT_LEN,
            self.DROPOUT).to(self.DEVICE)
        # self.criterion = nn.NLLLoss()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def fit(self, train_data, test_data):

        loss_trace = []
        loss_trace_test = []
        for epoch in tqdm(range(self.NUM_EPOCHS)):
            current_loss_train = 0
            train_loss_count = 0
            for i, (x, y) in enumerate(train_data):

                # Train
                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                outputs = self.model(x, y)
                outputs = outputs.permute(1, 2, 0)

                # Needed by self.criterion = nn.MSELoss
                # outputs = torch.argmax(outputs, dim=1)
                # outputs = outputs.type(torch.DoubleTensor)
                # outputs.requires_grad = True
                # y = y.type(torch.DoubleTensor)
                # y.requires_grad = True

                loss = self.criterion(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss_train += loss.item()
                train_loss_count += 1
                # print(f'Iteration {i} Training Loss: {loss.item()}')
            current_loss_train = current_loss_train / train_loss_count

            # Test
            current_loss_test = 0
            test_loss_count = 0
            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(test_data):
                    x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                    outputs = self.model(x, y)
                    outputs = outputs.permute(1, 2, 0)

                    loss = self.criterion(outputs, y)
                    current_loss_test += loss.item()
                    test_loss_count += 1
                current_loss_test = current_loss_test / test_loss_count
            self.model.train()

            # x_test, y_test = test_data.dataset.source, test_data.dataset.target
            # for j in range(len(x_test)):
            # self.model.eval()
            # with torch.no_grad():
            #     for j in range(3):
            #         y_pred_ = [0] + self.generate_fixed_length(torch.LongTensor([list(x_test[j])])) + [1]
            #         y_pred_ = np.array([n * -1 for n in y_pred_])
            #         y_pred = np.zeros((y_pred_.size, self.OUTPUT_VOCAB_SIZE))
            #         y_pred[np.arange(y_pred_.size), y_pred_] = 1
            #         current_loss_test += self.criterion(torch.FloatTensor(y_pred), torch.LongTensor(y_test[j])).item()
            # self.model.train()

            loss_trace.append(current_loss_train)
            loss_trace_test.append(current_loss_test)
            # print(f'Epoch {epoch} Training Loss: {current_loss_train}')
            # print(f'Epoch {epoch} Testing  Loss: {current_loss_test}')
            # f = open("loss-train.txt", "a")
            # f.write(f'{current_loss_train}\n')
            # f.close()
            # f = open("loss-test.txt", "a")
            # f.write(f'{current_loss_test}\n')
            # f.close()

        # loss curve
        # plt.plot(range(1, self.NUM_EPOCHS + 1), [n/max(loss_trace) for n in loss_trace], 'r-')
        # plt.plot(range(1, self.NUM_EPOCHS + 1), [n/max(loss_trace_test) for n in loss_trace_test], 'b-')
        # plt.title("Transformer Loss")
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.show()
        # plt.savefig(f'transformer-loss.png')
        # plt.clf()
        # plt.close()

        return loss_trace_test[-1]

