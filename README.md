# Bitcoin Price Prediction with Deep Learning

A full-stack application for predicting Bitcoin prices using deep learning models, with interactive visualizations and trading strategy backtesting.

## Project Overview

This project combines deep learning models with a modern web interface to:

1. Predict Bitcoin prices using LSTM, GRU, and Transformer models
2. Visualize price trends and predictions
3. Compare model performance
4. Backtest trading strategies based on model predictions

## Project Structure

- `backend/`: FastAPI-based Python backend
  - Deep learning models (LSTM, GRU, Transformer)
  - Data processing utilities
  - API endpoints
  - Backtesting engine
  
- `frontend/`: React-based web interface
  - Interactive dashboards
  - Price charts
  - Model comparison tools
  - Backtesting visualization

## Technologies Used

### Backend
- Python 3.8+
- FastAPI
- TensorFlow/Keras
- pandas, numpy, scikit-learn
- yfinance (for data fetching)
- Optuna (for hyperparameter tuning)

### Frontend
- React
- Tailwind CSS
- Chart.js
- React Router

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn

### Option 1: Running with Script (Recommended)

We've provided scripts to run both the backend and frontend concurrently:

#### Windows (PowerShell):
```
.\run-dev.ps1
```

#### Windows (Command Prompt):
```
run-dev.bat
```

### Option 2: Running with npm (Alternative)

1. Install dependencies in the root directory:
   ```
   npm install
   ```

2. Run the development servers:
   ```
   npm run dev
   ```

### Option 3: Manual Setup

#### Backend Setup
1. Navigate to the backend directory
2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend server
   ```bash
   python main.py
   ```

#### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies
   ```bash
   npm install
   ```
3. Create a `.env` file with
   ```
   VITE_API_URL=http://localhost:8000/api
   ```
4. Start the development server
   ```bash
   npm run dev
   ```

## Features

- **Price Prediction**: Forecast Bitcoin prices using multiple deep learning models
- **Model Comparison**: Compare performance metrics between different models
- **Backtesting**: Test trading strategies based on model predictions
- **Interactive Visualization**: View and analyze price trends and predictions
- **API Integration**: Connect custom applications to the prediction API

## Troubleshooting

If you encounter issues with Tailwind CSS, try reinstalling the dependencies:

```bash
cd frontend
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@3.3.3 postcss@8.4.29 autoprefixer@10.4.15
npx tailwindcss init -p
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Chart.js](https://www.chartjs.org/) 