import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from med_dataset import MedDataset
from data_loader import get_data_loaders


class PositionalEncoding1D(nn.Module):
    # Implementation based on "Attention is All You Need":
    #   PE(p, 2i)   = sin( p/(10000^(2i/d)) )
    #   PE(p, 2i+1) = cos( p/(10000^(2i/d)) )
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        if d_model % 2 == 1:
            pe = torch.zeros(max_len, d_model+1)
            
        # Create indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Scaling factors
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sin and cos to rows even and odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Adjust shape for broadcasting
        if d_model % 2 == 1:
            pe = pe[:, :-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    # Extending the 1D case to 2D:
    #   PE(p_x, p_y, 2i)   = sin( p_x/(10000^(2i/d)) )
    #   PE(p_x, p_y, 2i+1) = cos( p_x/(10000^(2i/d)) )
    #   PE(p_x, p_y, 2i+2) = sin( p_y/(10000^(2i/d)) )
    #   PE(p_x, p_y, 2i+3) = cos( p_y/(10000^(2i/d)) )
    def __init__(self, d_model, dropout=0.1, max_len=5000, max_width=5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, max_width, d_model)
        # Create row and column indices
        row_positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: (max_len, 1)
        col_positions = torch.arange(0, max_width, dtype=torch.float).unsqueeze(0)  # Shape: (1, max_width)
        # Scaling factors for rows and cols
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # Apply sin and cos to rows (even)
        pe[:, :, 0::2] = torch.sin(row_positions * div_term)
        pe[:, :, 1::2] = torch.cos(row_positions * div_term)
        # Apply sin and cos to cols (odd)
        pe[:, :, 2::2] = torch.sin(col_positions * div_term)
        pe[:, :, 3::2] = torch.cos(col_positions * div_term)
        # Adjust shape for broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1), :]
        return self.dropout(x)

class Transformer(nn.Module):
    # Adapted from: https://www.linkedin.com/pulse/llm-foundations-constructing-training-decoder-only-from-zebrowski-h0iac
    def __init__(self,
                 embedding_dim_out,
                 hidden_size_decoder,
                 n_heads_decoder,
                 n_layers_decoder,
                 max_tgt_len,
                 dropout_decoder,
                 output_sequence_length,
                 learning_rate,
                 batch_size,
                 num_epochs,
                 device):
        super(Transformer, self).__init__()
        # Hyper-parameters
        self.embedding_dim_out = embedding_dim_out      # embedding_dim_out = num features 
        self.hidden_size_decoder = hidden_size_decoder
        self.n_heads_decoder = n_heads_decoder
        self.n_layers_decoder = n_layers_decoder
        self.max_tgt_len = max_tgt_len
        self.dropout_decoder = dropout_decoder
        self.output_sequence_length = output_sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Embedding: Has same effect dimension-wise as nn.Embedding (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

        
        self.softmax_src = torch.nn.Softmax(dim=0)
        # Positional encoding layers
        self.dec_pe = PositionalEncoding1D(embedding_dim_out)
        # Encoder/decoder layers
        dec_layer = nn.TransformerDecoderLayer(embedding_dim_out, n_heads_decoder, hidden_size_decoder, dropout_decoder, dtype=torch.float64)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_decoder)
       
        # Final dense layer
        self.dense = nn.Linear(embedding_dim_out, output_sequence_length, dtype=torch.float64)
        self.final = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(12,1), padding=0, dtype=torch.float64)
        self.softmax_tgt = nn.Softmax(dim=0)
        
        # Criterion/optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Hardware device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, src, tgt):
        # print('forward')
        # print(f'src shape: {src.shape}')
        # print(f'tgt shape: {tgt.shape}')
        
        embed = self.softmax_src(src)  # Embedding: Normalize each feature at each time interval to 1 (softmax)
        embed = embed * math.sqrt(self.embedding_dim_out)
        
        pos_enc = self.dec_pe(embed)
        # print(f'pos_enc shape: {pos_enc.shape}')
        
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(self.device)
        # print(f'src_mask shape: {src_mask.shape}')
        
        transformer_out = self.decoder(tgt=pos_enc, memory=pos_enc, tgt_mask=src_mask)
        # print(f'transformer_out shape: {transformer_out.shape}')
        
        dense_out = self.dense(transformer_out)
        # print(f'dense_out shape: {dense_out.shape}')

        dense_out = torch.unsqueeze(dense_out, 1)
        # print(f'dense_out shape: {dense_out.shape}')
        final_out = self.final(dense_out)
        final_out = torch.squeeze(final_out, (1,2))
        # print(f'final_out shape: {final_out.shape}')
        
        final_out = self.softmax_tgt(final_out)
        return final_out


class TransformerModel:
    def __init__(self, hyperparams):
        DEVICE = 'cpu'
        self.device = torch.device(DEVICE)
        self.model = Transformer(hyperparams['NUM_INPUT_SEQUENCES'],
                                 hyperparams['DECODER_HIDDEN_SIZE'],
                                 hyperparams['NUM_HEADS_DECODER'],
                                 hyperparams['NUM_LAYERS_DECODER'],
                                 hyperparams['MAX_TGT_LEN'],
                                 hyperparams['DROPOUT_DECODER'],
                                 hyperparams['OUTPUT_SEQUENCE_LEN'],
                                 hyperparams['LEARNING_RATE'],
                                 hyperparams['BATCH_SIZE'],
                                 hyperparams['NUM_EPOCHS'],
                                 DEVICE).to(self.device)
    
    def fit(self, train_loader, test_loader):
        # Add start-of-sequence and end-of-sequence tokens
        # seq_start_train = pd.Series([0] * train_y.shape[0], name='seq_start')
        # seq_stop_train = pd.Series([1] * train_y.shape[0], name='seq_stop')
        # train_y.insert(0, 'seq_start', seq_start_train)
        # train_y.insert(train_y.shape[1], 'seq_stop', seq_stop_train)
        # seq_start_test = pd.Series([0] * test_y.shape[0], name='seq_start')
        # seq_stop_test = pd.Series([1] * test_y.shape[0], name='seq_stop')
        # test_y.insert(0, 'seq_start', seq_start_test)
        # test_y.insert(test_y.shape[1], 'seq_stop', seq_stop_test)
     
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
                outputs = self.model(x, y)
                loss = self.model.criterion(outputs, y)
                self.model.optimizer.zero_grad()
                loss.backward()
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
                    outputs = self.model(x, y)
                    outputs = torch.squeeze(outputs)
                    loss = self.model.criterion(outputs, y)
                    current_loss_test += loss.item()
            self.model.train()
            current_loss_test /= i
            loss_trace_test.append(current_loss_test)
        
        # Plot loss curve
        plt.figure()
        plt.plot(range(1, self.model.num_epochs + 1), loss_trace_train, 'r-', label='Train Loss')
        plt.plot(range(1, self.model.num_epochs + 1), loss_trace_test, 'b-', label='Test Loss')
        plt.title('Transformer Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig(f'loss_curve_transformer.png')


def main():
    file_path = "S3_Path Placeholder"
    train_loader, test_loader = get_data_loaders(file_path)
    
    hyperparams = {'NUM_INPUT_SEQUENCES': 29,
                   'DECODER_HIDDEN_SIZE': 1,
                     'NUM_HEADS_DECODER': 1,
                    'NUM_LAYERS_DECODER': 1,
                           'MAX_TGT_LEN': 4,
                       'DROPOUT_DECODER': 0,
                   'OUTPUT_SEQUENCE_LEN': 4,
                         'LEARNING_RATE': 0.003,
                            'BATCH_SIZE': 32,
                          'NUM_EPOCHS': 100}
    
    transformer_model = TransformerModel(hyperparams)
    transformer_model.fit(train_loader, test_loader)


if __name__ == "__main__":
    main()