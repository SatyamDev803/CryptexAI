# Bitcoin Price Prediction Backend

This is the backend component of the Bitcoin Price Prediction project. It provides ML models and APIs for predicting Bitcoin prices and backtesting trading strategies.

## Features

- Multiple deep learning models (LSTM, GRU)
- Hyperparameter tuning with Optuna
- Backtesting capabilities
- RESTful API with FastAPI
- Real-time data fetching with yfinance

## Project Structure

- `src/`: Source code
  - `models/`: ML model implementations
  - `utils/`: Utility functions
  - `api/`: API endpoints
  - `backtesting/`: Backtesting functionality
- `data/`: Data storage
- `models/`: Trained model storage
- `notebooks/`: Jupyter notebooks

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository
2. Navigate to the backend directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Server

```bash
python main.py
```

For development with auto-reload:

```bash
python main.py --reload
```

### API Documentation

Once the server is running, you can access the API documentation at:

```
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /api/health`: Check API health
- `GET /api/models`: List available models
- `GET /api/data/latest`: Get latest Bitcoin price data
- `GET /api/predict/{model_type}`: Get price predictions
- `GET /api/backtest/{model_type}`: Run backtests

## Model Training

To train new models, use the provided Jupyter notebooks or run:

```bash
python src/train.py --model lstm
```

## License

MIT 