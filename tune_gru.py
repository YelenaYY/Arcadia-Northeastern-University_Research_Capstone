import sys
import logging
from datetime import datetime
import optuna
from gru_ml import GruML
from data_loader import get_data_loaders


def objective(trial):

    # File path and column definitions
    S3_PATH = "Place Holder"
    file_path = S3_PATH
    binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
    target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']

    GRU_HIDDEN_SIZE = trial.suggest_int('GRU_HIDDEN_SIZE', 32, 128)
    GRU_NUM_LAYERS = trial.suggest_int('GRU_NUM_LAYERS', 1, 16)
    GRU_DROPOUT = trial.suggest_float('GRU_DROPOUT', 0.1, 0.5, step=0.1)
    DENSE_DROPOUT = trial.suggest_float('DENSE_DROPOUT', 0.1, 0.5, step=0.1)
    DENSE_HIDDEN_SIZE = trial.suggest_int('DENSE_HIDDEN_SIZE', 64, 128)
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 128, 512, step=64)
    LEARNING_RATE = trial.suggest_float('LEARNING_RATE', 0.00005, 0.001, log=True)
    NUM_EPOCHS = trial.suggest_int('NUM_EPOCHS', 20, 100)
    
    hyper_params = {
        'GRU_HIDDEN_SIZE': GRU_HIDDEN_SIZE,
        'GRU_NUM_LAYERS': GRU_NUM_LAYERS,
        'GRU_DROPOUT': GRU_DROPOUT,
        'DENSE_DROPOUT': DENSE_DROPOUT,
        'DENSE_HIDDEN_SIZE': DENSE_HIDDEN_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_EPOCHS': NUM_EPOCHS,
    }
    gru = GruML(hyper_params=hyper_params)
    train_loader, test_loader = get_data_loaders(file_path, binary_cols, target_cols, batch_size=BATCH_SIZE, split_ratio=0.8)
    score = gru.fit(train_loader, test_loader)
    return score


def tune():
    study_name = 'gru_' + datetime.today().strftime('%Y%m%d-%H%M%S')
    sqlite_storage = 'sqlite:///{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=sqlite_storage)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    tune()