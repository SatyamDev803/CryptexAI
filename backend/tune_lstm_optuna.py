#!/usr/bin/env python
"""
Script for hyperparameter tuning LSTMModel using Optuna.
"""
import argparse
import json
import optuna
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam
import os

from src.utils.data_loader import DataLoader
from src.models.lstm_model import LSTMModel


def objective(trial, symbol: str, future_day: int):
    # Hyperparameter suggestions
    lookback = trial.suggest_categorical('lookback', [30, 60, 90])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = [trial.suggest_int(f'units_l{i}', 16, 256, log=True) for i in range(n_layers)]
    dropouts = [trial.suggest_float(f'dropout_l{i}', 0.0, 0.5) for i in range(n_layers)]
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Load and prepare data
    loader = DataLoader(symbol=symbol)
    loader.fetch_data()
    data = loader.prepare_data(prediction_days=lookback, future_day=future_day)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']

    # Build and train model
    model = LSTMModel(
        lstm_units=units,
        dropout_rates=dropouts,
        optimizer=Adam(learning_rate=lr),
        input_shape=(lookback, X_train.shape[2])
    )
    model.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_test_inv = loader.inverse_transform(y_test)
    y_pred_inv = loader.inverse_transform(y_pred)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    return mape


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for LSTMModel")
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Ticker symbol')
    parser.add_argument('--future_day', type=int, default=1, help='Days ahead to predict')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args.symbol, args.future_day), n_trials=args.trials)

    print('Best MAPE:', study.best_value)
    print('Params:', study.best_trial.params)

    # Save best params
    out = {'symbol': args.symbol, 'future_day': args.future_day,
           'best_mape': float(study.best_value), 'params': study.best_trial.params}
    # Save into optuna_tune directory
    output_dir = os.path.join(os.path.dirname(__file__), "optuna_tune")
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{args.symbol}_lstm_optuna_best.json")
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved best hyperparameters to {fname}")


if __name__ == '__main__':
    main()
