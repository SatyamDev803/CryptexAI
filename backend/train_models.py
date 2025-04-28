#!/usr/bin/env python
"""
Script to train and save LSTM, GRU, and Transformer models.
"""
import os
import argparse
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.optimizers import Adam
from src.utils.data_loader import DataLoader
from src.models.lstm_model import LSTMModel
from src.models.gru_model import GRUModel
from src.models.transformer_model import TransformerModel

# Define project root and models directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def train_and_save(model_type: str, data_loader: DataLoader, epochs: int = 50, batch_size: int = 32, future_day: int = 1, hit_threshold: float = 0.02):
    print(f"Training {model_type} model...")
    # Fetch and prepare data
    df = data_loader.fetch_data()
    data = data_loader.prepare_data(future_day=future_day)
    X_train = data["x_train"]
    y_train = data["y_train"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    # Instantiate model
    if model_type.lower() == "lstm":
        model = LSTMModel(input_shape=input_shape)
    elif model_type.lower() == "gru":
        model = GRUModel(input_shape=input_shape)
    elif model_type.lower() == "transformer":
        model = TransformerModel(input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Train model
    history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save model file with symbol prefix for per-symbol loading
    path = os.path.join(MODELS_DIR, f"{data_loader.symbol}_{model_type.lower()}_model.keras")
    model.save(path)
    print(f"Saved {model_type} model for {data_loader.symbol} to {path}")
    # Compute metrics on hold-out test set and save to JSON
    x_test = data["x_test"]
    y_true = data["actual_prices"]
    y_pred_scaled = model.predict(x_test)
    y_pred = data_loader.inverse_transform(np.array(y_pred_scaled).flatten())
    # Ensure 1D arrays for metrics
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    # Directional accuracy
    accuracy = None
    if y_true.size > 1 and y_pred.size > 1:
        dirs_true = np.sign(np.diff(y_true))
        dirs_pred = np.sign(np.diff(y_pred))
        min_len = min(dirs_true.size, dirs_pred.size)
        if min_len > 0:
            accuracy = float((dirs_true[:min_len] == dirs_pred[:min_len]).mean() * 100)
    # Compute hit rate: % of predictions within ±hit_threshold of true price
    hit_rate = float(np.mean(np.abs((y_pred - y_true) / y_true) <= hit_threshold) * 100)
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "accuracy": accuracy,
        "hit_rate": hit_rate
    }
    metrics_path = os.path.join(MODELS_DIR, f"{data_loader.symbol}_{model_type.lower()}_metrics.json")
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf)
    print(f"Saved metrics to {metrics_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for given crypto symbols")
    parser.add_argument("--symbols", type=str,
                        default="BTC-USD,ETH-USD,SOL-USD,ADA-USD,XRP-USD",
                        help="Comma-separated list of ticker symbols")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--future_day", type=int, default=1, help="Days ahead to predict (1 for next-day forecasts)")
    parser.add_argument("--hit_threshold", type=float, default=0.02, help="Hit rate threshold as decimal (e.g. 0.02 for ±2%)")
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    # Directory where Optuna tuning JSONs are saved
    config_dir = os.path.join(os.path.dirname(__file__), "optuna_tune")
    for sym in symbols:
        loader = DataLoader(symbol=sym)
        for m in ["lstm", "gru", "transformer"]:
            # Check for tuned params file
            cfg_path = os.path.join(config_dir, f"{sym}_{m}_optuna_best.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as cf:
                    cfg = json.load(cf)
                params = cfg["params"]
                lookback = params.get("lookback", 30)
                loader.fetch_data()
                data = loader.prepare_data(prediction_days=lookback, future_day=args.future_day)
                X_train, y_train = data["x_train"], data["y_train"]
                input_shape = (X_train.shape[1], X_train.shape[2])
                batch_size, epochs, lr = params.get("batch_size", args.batch_size), params.get("epochs", args.epochs), params.get("lr", 1e-3)
                # Instantiate tuned model
                if m == "lstm":
                    units = [params[f"units_l{i}"] for i in range(params.get("n_layers",1))]
                    dropouts = [params[f"dropout_l{i}"] for i in range(params.get("n_layers",1))]
                    model = LSTMModel(lstm_units=units, dropout_rates=dropouts,
                                      optimizer=Adam(learning_rate=lr), input_shape=input_shape)
                elif m == "gru":
                    units = [params[f"units_l{i}"] for i in range(params.get("n_layers",1))]
                    dropouts = [params[f"dropout_l{i}"] for i in range(params.get("n_layers",1))]
                    model = GRUModel(gru_units=units, dropout_rates=dropouts,
                                     optimizer=Adam(learning_rate=lr), input_shape=input_shape)
                else:
                    model = TransformerModel(num_layers=params.get("n_layers",2),
                                             d_model=params.get("d_model",64),
                                             num_heads=params.get("num_heads",4),
                                             dff=params.get("dff",128),
                                             dropout_rate=params.get("dropout_rate",0.1),
                                             optimizer=Adam(learning_rate=lr), input_shape=input_shape)
                hist = model.train(X_train, y_train, epochs=epochs,
                                   batch_size=batch_size, validation_split=0.1)
                # Save tuned model
                os.makedirs(MODELS_DIR, exist_ok=True)
                model_path = os.path.join(MODELS_DIR, f"{sym}_{m}_model.keras")
                model.save(model_path)
                print(f"Saved tuned {m.upper()} model for {sym} to {model_path}")
                # Compute & save metrics including hit_rate
                x_test = data["x_test"]
                y_true = np.array(data.get("actual_prices", [])).flatten()
                y_pred = loader.inverse_transform(np.array(model.predict(x_test)).flatten())
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100
                accuracy = None
                if y_true.size > 1:
                    dirs_true = np.sign(np.diff(y_true)); dirs_pred = np.sign(np.diff(y_pred))
                    accuracy = float((dirs_true[:min(len(dirs_true), len(dirs_pred))] == dirs_pred[:min(len(dirs_true), len(dirs_pred))]).mean() * 100)
                # Compute hit rate with threshold
                hit_rate = float(np.mean(np.abs((y_pred - y_true) / y_true) <= args.hit_threshold) * 100)
                metrics = {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "accuracy": accuracy, "hit_rate": hit_rate}
                metrics_path = os.path.join(MODELS_DIR, f"{sym}_{m}_metrics.json")
                with open(metrics_path, "w") as mf:
                    json.dump(metrics, mf)
                print(f"Saved metrics to {metrics_path}")
            else:
                hist = train_and_save(m, loader, epochs=args.epochs, batch_size=args.batch_size, future_day=args.future_day, hit_threshold=args.hit_threshold)
            print(f"{m.upper()} for {sym} completed: epochs={hist.get('epochs_completed')}, final_loss={hist.get('final_loss')}")
