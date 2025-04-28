import React, { createContext, useContext, useState, useEffect } from 'react';
import api from '../services/api';

// Available cryptocurrencies
const cryptocurrencies = [
  { id: 'bitcoin', name: 'Bitcoin', symbol: 'BTC' },
  { id: 'ethereum', name: 'Ethereum', symbol: 'ETH' },
  { id: 'solana', name: 'Solana', symbol: 'SOL' },
  { id: 'cardano', name: 'Cardano', symbol: 'ADA' },
  { id: 'ripple', name: 'XRP', symbol: 'XRP' }
];

// Create context
const AppContext = createContext();

// Custom hook to use the context
export const useAppContext = () => useContext(AppContext);

export function AppProvider({ children }) {
  // App state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cryptoPrice, setCryptoPrice] = useState(null);
  const [priceHistory, setPriceHistory] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [darkMode, setDarkMode] = useState(() => {
    // Check for saved preference or use system preference
    const saved = localStorage.getItem('darkMode');
    if (saved !== null) {
      return JSON.parse(saved);
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });
  const [selectedCrypto, setSelectedCrypto] = useState(cryptocurrencies[0]);
  const [predictionData, setPredictionData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);

  // Load initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        
        // Check API health
        try {
          await api.checkHealth();
          console.log('API health check passed');
        } catch (healthError) {
          console.warn('API health check failed, using mock data:', healthError.message);
        }
        
        // Get available models
        try {
          const models = await api.getModels();
          setAvailableModels(models);
          console.log('Models loaded:', models);
        } catch (modelsError) {
          console.warn('Failed to load models:', modelsError.message);
          setAvailableModels(['lstm', 'gru', 'transformer']);
        }
        
        // Get latest price data for selected crypto
        await refreshPriceData();
        
        setLoading(false);
      } catch (err) {
        console.error('Error during initial data load:', err);
        setError(`Failed to initialize app: ${err.message}`);
        setLoading(false);
      }
    };
    
    fetchInitialData();
  }, []);

  // Refresh data when crypto changes
  useEffect(() => {
    if (!loading) {
      refreshPriceData();
      // Clear previous prediction and backtest data
      setPredictionData(null);
      setBacktestData(null);
    }
  }, [selectedCrypto.id]);

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(prev => {
      const newValue = !prev;
      localStorage.setItem('darkMode', JSON.stringify(newValue));
      return newValue;
    });
  };

  // Apply dark mode to HTML element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Initialize dark mode on first load
  useEffect(() => {
    // Apply dark mode on initial load
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  // API functions
  const getPredictions = async (modelType = selectedModel, days = 30, cryptoId = selectedCrypto.id) => {
    try {
      setLoading(true);
      // Allow overriding the default selected model
      const model = modelType || selectedModel;
      const crypto = cryptoId || selectedCrypto.id;
      
      const data = await api.getPredictions(model, days, crypto);
      
      // Only update predictionData if querying for the currently selected crypto
      if (crypto === selectedCrypto.id) {
      setPredictionData(data);
      }
      
      setLoading(false);
      return data;
    } catch (err) {
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  const runBacktest = async (days = 365, threshold = 0.01, initialBalance = 10000) => {
    try {
      setLoading(true);
      const data = await api.runBacktest(selectedModel, days, threshold, initialBalance, selectedCrypto.id);
      setBacktestData(data);
      setLoading(false);
      return data;
    } catch (err) {
      setError(err.message);
      setLoading(false);
      throw err;
    }
  };

  const refreshPriceData = async (days = 30) => {
    try {
      setLoading(true);
      console.log(`Fetching price data for ${selectedCrypto.id} for ${days} days`);
      const data = await api.getLatestData(days, selectedCrypto.id);
      
      if (data && data.prices && data.prices.length > 0) {
        setPriceHistory(data.prices);
        // Set the crypto price from the most recent data point
        setCryptoPrice(data.prices[data.prices.length - 1].close);
        console.log(`Successfully loaded ${data.prices.length} price points`);
      } else {
        console.warn('Price data response was empty or malformed:', data);
        // Set realistic fallback prices if API returns empty data
        const fallbackPrices = getRealisticFallbackPrice(selectedCrypto.id);
        setCryptoPrice(fallbackPrices);
      }
      
      setLoading(false);
      return data;
    } catch (err) {
      console.error('Error refreshing price data:', err);
      setError(`Failed to fetch price data: ${err.message}`);
      // Set realistic fallback prices if API fails
      const fallbackPrices = getRealisticFallbackPrice(selectedCrypto.id);
      setCryptoPrice(fallbackPrices);
      setLoading(false);
    }
  };

  // Helper function to get realistic fallback prices
  const getRealisticFallbackPrice = (cryptoId) => {
    switch(cryptoId) {
      case 'bitcoin':
        return 84000;
      case 'ethereum':
        return 3000;
      case 'solana':
        return 150;
      case 'cardano':
        return 0.45;
      case 'ripple':
        return 0.50;
      default:
        return 100;
    }
  };

  // Change selected cryptocurrency
  const changeCrypto = (cryptoId) => {
    const crypto = cryptocurrencies.find(c => c.id === cryptoId);
    if (crypto) {
      setSelectedCrypto(crypto);
    }
  };

  // Provide context value
  const contextValue = {
    loading,
    setLoading,
    error,
    setError,
    cryptoPrice,
    setCryptoPrice,
    priceHistory,
    setPriceHistory,
    availableModels,
    selectedModel,
    setSelectedModel,
    darkMode,
    toggleDarkMode,
    cryptocurrencies,
    selectedCrypto,
    changeCrypto,
    getPredictions,
    runBacktest,
    refreshPriceData,
    predictionData,
    backtestData,
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
}

export default AppContext; 