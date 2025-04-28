const API_URL = import.meta.env.VITE_API_URL || 'https://cryptexai.onrender.com';

/**
 * Generic request function
 */
async function request(endpoint, options = {}) {
  const url = `${API_URL}${endpoint}`;
  
  console.log(`API Request: ${url}`);
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      let errorDetails;
      try {
        errorDetails = await response.json();
      } catch {
        errorDetails = { 
          message: `API endpoint ${endpoint} returned ${response.status}: ${response.statusText}` 
        };
      }
      
      const errorMessage = errorDetails.message || errorDetails.detail || `API endpoint ${endpoint} not available`;
      console.error(`API Error (${response.status}): ${errorMessage}`);
      throw new Error(errorMessage);
    }
    
    const data = await response.json();
    console.log(`API Response from ${endpoint}:`, data);
    return data;
  } catch (error) {
    console.error(`API request to ${endpoint} failed:`, error);
    throw error;
  }
}

/**
 * Health check
 */
export const checkHealth = () => request('/health');

/**
 * Fetch available models
 */
export const getModels = () => request('/models');

/**
 * Fetch latest cryptocurrency price data
 * @param {number} days - Number of days of data to return
 * @param {string} symbol - Symbol of the cryptocurrency (e.g., 'BTC-USD', 'ETH-USD')
 */
export const getLatestData = async (days = 30, symbol = 'BTC-USD') => {
  try {
    return await request(`/data/latest?days=${days}&symbol=${symbol}`);
  } catch (error) {
    console.warn('Using mock data for getLatestData:', error.message);
    return generateMockData.priceHistory(symbol, days);
  }
};

/**
 * Get price predictions
 * @param {string} modelType - Model type (lstm, gru, etc.)
 * @param {number} days - Number of days to predict
 * @param {string} symbol - Symbol of the cryptocurrency (e.g., 'BTC-USD', 'ETH-USD')
 */
export const getPredictions = async (modelType, days = 30, symbol = 'BTC-USD') => {
  try {
    return await request(`/predict/${modelType}?days=${days}&symbol=${symbol}`);
  } catch (error) {
    console.warn('Using mock data for getPredictions:', error.message);
    return generateMockData.predictions(symbol, days, modelType);
  }
};

/**
 * Run a backtest
 * @param {string} modelType - Model type (lstm, gru, etc.)
 * @param {number} days - Number of days for backtesting
 * @param {number} threshold - Price change threshold for trade signals
 * @param {number} initialBalance - Initial balance
 * @param {string} symbol - Symbol of the cryptocurrency (e.g., 'BTC-USD', 'ETH-USD')
 */
export const runBacktest = async (
  modelType, 
  days = 365, 
  threshold = 0.01, 
  initialBalance = 10000,
  symbol = 'BTC-USD'
) => {
  const url = `/backtest/${modelType}?days=${days}&threshold=${threshold}&initial_balance=${initialBalance}&symbol=${symbol}`;
  console.log(`Sending real backtest request to: ${url}`);
  
  // Send the actual request without fallback to mock data
  return await request(url);
};

/**
 * Get model comparison data
 * @param {string} symbol - Symbol of the cryptocurrency (e.g., 'BTC-USD', 'ETH-USD')
 * @param {number} days - Number of days to predict
 */
export const modelComparison = async (symbol = 'BTC-USD', days = 14) => {
  try {
    return await request(`/compare-models?days=${days}&symbol=${symbol}`);
  } catch (error) {
    console.warn('Using mock data for modelComparison:', error.message);
    return generateMockData.modelComparison(symbol, days);
  }
};

/**
 * Mock data generation for development/demonstration
 */
export const generateMockData = {
  // Mock price history for the last 30 days
  priceHistory: (symbol = 'BTC-USD', days = 30) => {
    const prices = [];
    const now = new Date();
    
    // More realistic base prices
    const basePrice = symbol.toLowerCase().includes('btc') || symbol === 'bitcoin' ? 84000 : 
                     symbol.toLowerCase().includes('eth') || symbol === 'ethereum' ? 3000 : 
                     symbol.toLowerCase().includes('sol') || symbol === 'solana' ? 150 : 
                     symbol.toLowerCase().includes('ada') || symbol === 'cardano' ? 0.45 : 
                     symbol.toLowerCase().includes('xrp') || symbol === 'ripple' ? 0.50 : 200;
    
    // Start from a more realistic current price
    let currentPrice = basePrice;
    
    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(now.getDate() - (days - i));
      
      // Generate some randomness to simulate price movement (smaller volatility for more stable values)
      const volatility = 0.01; // 1% daily volatility
      const changePercent = (Math.random() - 0.5) * volatility * 2;
      currentPrice = currentPrice * (1 + changePercent);
      
      prices.push({
        date: date.toISOString().split('T')[0],
        open: currentPrice * 0.99,
        high: currentPrice * 1.02,
        low: currentPrice * 0.98,
        close: currentPrice,
        volume: Math.floor(Math.random() * 10000) + 1000
      });
    }
    
    return { prices };
  },
  
  // Mock prediction data
  predictions: (symbol = 'BTC-USD', days = 7, modelType = 'lstm') => {
    const predictions = [];
    const now = new Date();
    
    // More realistic base prices
    const basePrice = symbol.toLowerCase().includes('btc') || symbol === 'bitcoin' ? 84000 : 
                     symbol.toLowerCase().includes('eth') || symbol === 'ethereum' ? 3000 : 
                     symbol.toLowerCase().includes('sol') || symbol === 'solana' ? 150 : 
                     symbol.toLowerCase().includes('ada') || symbol === 'cardano' ? 0.45 : 
                     symbol.toLowerCase().includes('xrp') || symbol === 'ripple' ? 0.50 : 200;
    
    // Add a small random variation to current price (±2%)
    const lastPrice = basePrice * (1 + (Math.random() - 0.5) * 0.04);
    let currentPrice = lastPrice;
    
    // Different trend patterns based on model
    let trendFactor;
    switch(modelType.toLowerCase()) {
      case 'gru':
        trendFactor = 0.001; // 0.1% daily trend
        break;
      case 'transformer':
        trendFactor = 0.002; // 0.2% daily trend
        break;
      case 'lstm':
      default:
        trendFactor = 0.0015; // 0.15% daily trend
    }
    
    for (let i = 1; i <= days; i++) {
      const date = new Date();
      date.setDate(now.getDate() + i);
      
      // Add trend with some randomness
      const randomness = (Math.random() - 0.5) * 0.01; // ±0.5% daily randomness
      // Use smaller trend factor to avoid extreme projections
      currentPrice = currentPrice * (1 + trendFactor + randomness);
      
      predictions.push({
        date: date.toISOString().split('T')[0],
        price: currentPrice,
        lower_bound: currentPrice * 0.98, // 2% lower bound
        upper_bound: currentPrice * 1.02  // 2% upper bound
      });
    }
    
    // Adjust metrics based on the model type
    let accuracy, mae, rmse, mape;
    switch(modelType.toLowerCase()) {
      case 'gru':
        accuracy = 68.3;
        mae = lastPrice * 0.015;
        rmse = lastPrice * 0.020;
        mape = 1.5;
        break;
      case 'transformer':
        accuracy = 75.8;
        mae = lastPrice * 0.012;
        rmse = lastPrice * 0.016;
        mape = 1.2;
        break;
      case 'lstm':
      default:
        accuracy = 72.5;
        mae = lastPrice * 0.014;
        rmse = lastPrice * 0.018;
        mape = 1.4;
    }
    
    return {
      symbol,
      model: modelType,
      current_price: lastPrice,
      predictions,
      metrics: {
        mae,
        rmse,
        mape,
        accuracy
      }
    };
  },
  
  // Mock backtest data
  backtest: (symbol = 'BTC-USD') => {
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);
    
    // Get the base price for this cryptocurrency
    const basePrice = symbol.toLowerCase().includes('btc') || symbol === 'bitcoin' ? 84000 : 
                     symbol.toLowerCase().includes('eth') || symbol === 'ethereum' ? 3000 : 
                     symbol.toLowerCase().includes('sol') || symbol === 'solana' ? 150 : 
                     symbol.toLowerCase().includes('ada') || symbol === 'cardano' ? 0.45 : 
                     symbol.toLowerCase().includes('xrp') || symbol === 'ripple' ? 0.50 : 200;
    
    // Generate mock trade history
    const trades = [];
    let currentPosition = "CASH";
    let cashBalance = 10000;
    let cryptoAmount = 0;
    
    for (let i = 0; i < 20; i++) {
      const tradeDate = new Date(startDate);
      tradeDate.setDate(tradeDate.getDate() + (i * 18)); // Spread across the year
      
      // Generate a realistic price with some randomness
      const price = basePrice * (1 + (Math.random() - 0.5) * 0.1);
      
      if (currentPosition === "CASH") {
        // Buy crypto
        cryptoAmount = cashBalance / price;
        trades.push({
          date: tradeDate.toISOString().split('T')[0],
          type: 'BUY',
          price: price,
          amount: cryptoAmount,
          balance_after: 0 // Now holding crypto instead of cash
        });
        currentPosition = "CRYPTO";
        cashBalance = 0;
      } else {
        // Sell crypto
        cashBalance = cryptoAmount * price;
        trades.push({
          date: tradeDate.toISOString().split('T')[0],
          type: 'SELL',
          price: price,
          amount: cryptoAmount,
          balance_after: cashBalance
        });
        currentPosition = "CASH";
        cryptoAmount = 0;
      }
    }
    
    // Generate balance history data
    const balance_history = [];
    let equity = 10000;
    for (let i = 0; i < 365; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Add some randomness with upward trend
      const change = (Math.random() - 0.4) * 100;
      equity += change;
      if (equity < 8000) equity = 8000 + Math.random() * 500;
      
      // Format as [date, value] array to match backend format
      balance_history.push([
        date.toISOString().split('T')[0],
        equity
      ]);
    }
    
    // Use final value from balance history for consistency
    const finalBalance = balance_history[balance_history.length - 1][1];
    const returns = ((finalBalance - 10000) / 10000) * 100;
    
    return {
      symbol,
      model: 'lstm',
      initial_balance: 10000,
      final_balance: finalBalance,
      returns: returns,
      sharpe_ratio: 1.85,
      max_drawdown: 12.34,
      trades_count: trades.length,
      win_rate: 68.75,
      trades: trades,
      balance_history: balance_history,
      position_history: balance_history.map((item, i) => [
        item[0], // date
        i % 2 === 0 ? 0 : 0.1 // alternating positions
      ])
    };
  },
  
  // Mock model comparison data
  modelComparison: (symbol = 'BTC-USD', days = 14) => {
    // Generate dates for predictions
    const dates = [];
    const now = new Date();
    for (let i = 1; i <= days; i++) {
      const date = new Date();
      date.setDate(now.getDate() + i);
      dates.push(date.toISOString().split('T')[0]);
    }
    
    // Base price based on cryptocurrency
    const basePrice = symbol === 'BTC-USD' ? 50000 : 
                    symbol === 'ETH-USD' ? 3000 : 
                    symbol === 'SOL-USD' ? 100 : 
                    symbol === 'ADA-USD' ? 1.2 : 
                    symbol === 'XRP-USD' ? 0.8 : 200;
    
    // Generate prediction data for each model
    const lstm = [], gru = [], transformer = [];
    
    for (let i = 0; i < days; i++) {
      // Add some trend and randomness for each model
      const trendFactor = 0.003; // 0.3% daily trend
      const day = i + 1;
      
      // Each model has slightly different predictions
      lstm.push(basePrice * (1 + (trendFactor * day) + (Math.random() - 0.5) * 0.02));
      gru.push(basePrice * (1 + (trendFactor * day * 0.9) + (Math.random() - 0.5) * 0.025));
      transformer.push(basePrice * (1 + (trendFactor * day * 1.1) + (Math.random() - 0.5) * 0.015));
    }
    
    return {
      symbol,
      predictions: {
        dates,
        lstm,
        gru,
        transformer
      },
      metrics: {
        lstm: {
          accuracy: 72.5,
          mae: 325.45,
          rmse: 428.91,
          time: 245
        },
        gru: {
          accuracy: 68.3,
          mae: 342.18,
          rmse: 456.32,
          time: 180
        },
        transformer: {
          accuracy: 75.8,
          mae: 298.76,
          rmse: 389.45,
          time: 310
        }
      },
      error_distribution: {
        bins: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        lstm: [2, 5, 10, 15, 25, 30, 25, 15, 10, 5, 2],
        gru: [3, 6, 12, 16, 23, 28, 23, 16, 12, 6, 3],
        transformer: [1, 4, 8, 14, 27, 32, 27, 14, 8, 4, 1]
      }
    };
  }
};

const api = {
  checkHealth,
  getModels,
  getLatestData,
  getPredictions,
  runBacktest,
  modelComparison,
  mock: generateMockData
};

export default api; 