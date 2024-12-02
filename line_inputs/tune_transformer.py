import sys
import logging
from datetime import datetime
import optuna
from transformer_ml import TransformerML
from data_loader import get_data_loaders


def objective(trial):

    # File path and column definitions
    S3_PATH = "Place Holder"
    file_path = S3_PATH
    binary_cols = ['male', 'heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']
    target_cols = ['heart_failure_1mo', 'heart_failure_12mo', 'heart_failure_120mo']

    NUM_EPOCHS = trial.suggest_int('NUM_EPOCHS', 20, 100)
    HIDDEN_SIZE = trial.suggest_int('HIDDEN_SIZE', 64, 512)
    EMBEDDING_DIM_FACTOR = trial.suggest_int('EMBEDDING_DIM_FACTOR', 2, 8, step=2)
    BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 128, 512, step=64)
    NUM_HEADS = trial.suggest_categorical('NUM_HEADS', [4, 8])
    NUM_LAYERS = trial.suggest_int('NUM_LAYERS', 2, 6)
    LEARNING_RATE = trial.suggest_float('LEARNING_RATE', 0.00005, 0.001, log=True)
    DROPOUT = trial.suggest_float('DROPOUT', 0.1, 0.5, step=0.1)
    hyper_params = {
        'NUM_EPOCHS': NUM_EPOCHS,
        'HIDDEN_SIZE': HIDDEN_SIZE,
        'EMBEDDING_DIM_FACTOR': EMBEDDING_DIM_FACTOR,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_HEADS': NUM_HEADS,
        'NUM_LAYERS': NUM_LAYERS,
        'LEARNING_RATE': LEARNING_RATE,
        'DROPOUT': DROPOUT,
    }
    trasformer = TransformerML(hyper_params=hyper_params)
    train_loader, test_loader = get_data_loaders(file_path, binary_cols, target_cols, batch_size=BATCH_SIZE, split_ratio=0.8)
    score = trasformer.fit(train_loader, test_loader)
    return score


def tune():
    study_name = 'transformer_' + datetime.today().strftime('%Y%m%d-%H%M%S')
    sqlite_storage = 'sqlite:///{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, direction='minimize', storage=sqlite_storage)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    tune()
