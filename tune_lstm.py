import sys
import logging
from datetime import datetime
import optuna
from lstm_ml import LstmML  # Import the LSTM-based model
from data_loader import get_data_loaders

def objective(trial):

    # File path and column definitions
    S3_PATH = "Place Holder"
    file_path = S3_PATH
    binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
    target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']

    LSTM_HIDDEN_SIZE = trial.suggest_int('LSTM_HIDDEN_SIZE', 32, 128)
    LSTM_NUM_LAYERS = trial.suggest_int('LSTM_NUM_LAYERS', 1, 16)
    LSTM_DROPOUT = trial.suggest_float('LSTM_DROPOUT', 0.1, 0.5, step=0.1)
    DENSE_DROPOUT = trial.suggest_float('DENSE_DROPOUT', 0.1, 0.5, step=0.1)
    DENSE_HIDDEN_SIZE = trial.suggest_int('DENSE_HIDDEN_SIZE', 64, 128)
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 128, 512, step=64)
    LEARNING_RATE = trial.suggest_float('LEARNING_RATE', 0.00005, 0.001, log=True)
    NUM_EPOCHS = trial.suggest_int('NUM_EPOCHS', 20, 100)
    
    hyper_params = {
        'LSTM_HIDDEN_SIZE': LSTM_HIDDEN_SIZE,
        'LSTM_NUM_LAYERS': LSTM_NUM_LAYERS,
        'LSTM_DROPOUT': LSTM_DROPOUT,
        'DENSE_DROPOUT': DENSE_DROPOUT,
        'DENSE_HIDDEN_SIZE': DENSE_HIDDEN_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_EPOCHS': NUM_EPOCHS,
    }
    lstm = LstmML(hyper_params=hyper_params)  # Instantiate LstmML instead of GruML
    train_loader, test_loader = get_data_loaders(file_path, binary_cols, target_cols, batch_size=BATCH_SIZE, split_ratio=0.8)
    score = lstm.fit(train_loader, test_loader)  # Use the LSTM model's fit method
    return score

def tune():
    study_name = 'lstm_' + datetime.today().strftime('%Y%m%d-%H%M%S')
    sqlite_storage = 'sqlite:///{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=sqlite_storage)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(objective, n_trials=100)

if __name__ == "__main__":
    tune()
