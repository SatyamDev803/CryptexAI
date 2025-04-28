#!/usr/bin/env python
"""
General Optuna tuning for LSTM, GRU, and Transformer models across multiple symbols.
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
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel


def objective(trial, model_type: str, symbol: str, future_day: int):
    # Shared hyperparameters
    lookback = trial.suggest_categorical('lookback', [30, 60, 90])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)
    units = [trial.suggest_int(f'units_l{i}', 16, 256, log=True) for i in range(n_layers)]
    dropouts = [trial.suggest_float(f'dropout_l{i}', 0.0, 0.5) for i in range(n_layers)]

    # Model-specific extra params for transformer
    if model_type == 'transformer':
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        num_heads = trial.suggest_int('num_heads', 1, 4)
        dff = trial.suggest_categorical('dff', [64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    # Load data
    loader = DataLoader(symbol=symbol)
    loader.fetch_data()
    data = loader.prepare_data(prediction_days=lookback, future_day=future_day)
    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']

    # Instantiate model
    if model_type == 'lstm':
        model = LSTMModel(lstm_units=units, dropout_rates=dropouts,
                          optimizer=Adam(learning_rate=lr),
                          input_shape=(lookback, X_train.shape[2]))
    elif model_type == 'gru':
        model = GRUModel(gru_units=units, dropout_rates=dropouts,
                         optimizer=Adam(learning_rate=lr),
                         input_shape=(lookback, X_train.shape[2]))
    elif model_type == 'transformer':
        model = TransformerModel(num_layers=n_layers, d_model=d_model,
                                 num_heads=num_heads, dff=dff,
                                 dropout_rate=dropout_rate,
                                 optimizer=Adam(learning_rate=lr),
                                 input_shape=(lookback, X_train.shape[2]))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train & evaluate
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_split=0.1, verbose=0)
    y_pred = model.predict(X_test)
    y_test_inv = loader.inverse_transform(y_test)
    y_pred_inv = loader.inverse_transform(y_pred)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    return mape


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for multiple models and symbols")
    parser.add_argument('--symbols', type=str, default='BTC-USD,ADA-USD',
                        help='Comma-separated list of symbols')
    parser.add_argument('--models', type=str, default='lstm,gru,transformer',
                        help='Comma-separated list of model types')
    parser.add_argument('--future_day', type=int, default=1,
                        help='Days ahead to predict')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of trials per study')
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]
    models = [m.strip() for m in args.models.split(',')]

    for symbol in symbols:
        for model_type in models:
            print(f"Tuning {model_type.upper()} for {symbol}...")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda t: objective(t, model_type, symbol, args.future_day),
                           n_trials=args.trials)

            out = {
                'model': model_type,
                'symbol': symbol,
                'future_day': args.future_day,
                'best_mape': float(study.best_value),
                'params': study.best_trial.params
            }
            # Save into optuna_tune directory
            output_dir = os.path.join(os.path.dirname(__file__), "optuna_tune")
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f"{symbol}_{model_type}_optuna_best.json")
            with open(fname, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"Saved best params to {fname}\n")


if __name__ == '__main__':
    main()
