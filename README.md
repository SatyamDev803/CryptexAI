# CryptexAI - Cryptocurrency Price Prediction Platform

A comprehensive full-stack application leveraging state-of-the-art deep learning models for cryptocurrency price prediction, interactive visualizations, and advanced trading strategy backtesting. Built with Python, FastAPI, TensorFlow, and React.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com/)
[![React 18+](https://img.shields.io/badge/react-18.0+-61dafb.svg)](https://reactjs.org/)

## Project Overview

CryptexAI is a sophisticated cryptocurrency prediction platform that combines cutting-edge deep learning architectures with an intuitive web interface. The platform offers:

1. Advanced price predictions for multiple cryptocurrencies (BTC, ETH, ADA, SOL, XRP) using:
   - LSTM (Long Short-Term Memory) networks
   - GRU (Gated Recurrent Unit) networks
   - Transformer models with attention mechanisms
2. Real-time price tracking and visualization with interactive charts
3. Comprehensive model performance metrics and comparisons
4. Advanced backtesting engine for strategy validation
5. API endpoints for seamless integration with external applications

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

### Quick Start (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/SatyamDev803/CryptexAI.git
cd CryptexAI
```

2. Install root dependencies:
```bash
npm install
```

3. Run both services:
```bash
npm run dev
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

### Model Features
- **Multi-Cryptocurrency Support**: Predictions for BTC, ETH, ADA, SOL, and XRP
- **Multiple Model Architectures**: 
  - LSTM for long-term dependency learning
  - GRU for efficient training and memory usage
  - Transformer models with self-attention mechanisms
- **Hyperparameter Optimization**: Automated model tuning using Optuna
- **Real-time Predictions**: Live price forecasting with configurable timeframes

### Trading Features
- **Advanced Backtesting Engine**: 
  - Custom strategy implementation
  - Performance metrics calculation
  - Risk assessment tools
- **Trading Signals**: 
  - Buy/Sell indicators based on model predictions
  - Confidence scores for each signal
  - Multiple timeframe analysis

### Visualization Features
- **Interactive Charts**: 
  - Real-time price updates
  - Technical indicators
  - Prediction overlays
- **Model Performance Dashboards**:
  - Accuracy metrics
  - Error analysis
  - Model comparison tools
- **Backtesting Results**:
  - Profit/Loss visualization
  - Trade history
  - Performance metrics

### API Features
- **RESTful Endpoints**: 
  - Price predictions
  - Model metrics
  - Historical data
- **WebSocket Support**: Real-time price and prediction updates
- **Authentication**: Secure API access with key management
- **Rate Limiting**: Controlled access to API resources

## Troubleshooting

### Tailwind CSS Issues
If you encounter styling issues:
```bash
cd frontend
npm uninstall tailwindcss postcss autoprefixer
npm install -D tailwindcss@3.3.3 postcss@8.4.29 autoprefixer@10.4.15
npx tailwindcss init -p
```

### Port Conflicts
- Frontend runs on port 5173 by default
- Backend runs on port 8000 by default
- Modify these in `frontend/vite.config.js` and `backend/main.py` respectively if needed

### Common Issues
1. **Model Loading Errors**:
   - Ensure you have enough RAM (8GB+ recommended)
   - Check if model files are properly downloaded
   
2. **Data Fetching Issues**:
   - Verify internet connection
   - Check yfinance API status
   
3. **Performance Issues**:
   - Consider reducing prediction window size
   - Optimize browser cache settings
   - Check system resources usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Chart.js](https://www.chartjs.org/) 