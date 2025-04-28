import { Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import CryptoSelector from './CryptoSelector';
import DarkModeToggle from './DarkModeToggle';

export default function Navbar({ onMenuClick }) {
  const location = useLocation();
  const { cryptoPrice, selectedCrypto, loading } = useAppContext();
  const [priceChange, setPriceChange] = useState({ value: 0, isPositive: true });
  
  // Mock price change as this would normally come from the API
  useEffect(() => {
    if (cryptoPrice) {
      // Generate a random change between -5% and +5%
      const changePercent = (Math.random() * 10 - 5).toFixed(2);
      setPriceChange({
        value: `${Math.abs(changePercent)}%`,
        isPositive: parseFloat(changePercent) >= 0
      });
    }
  }, [cryptoPrice]);
  
  const formatPrice = (price) => {
    if (!price) return 'Loading...';
    
    // Format based on price magnitude
    if (price > 1000) {
      return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    } else if (price > 1) {
      return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else {
      return `$${price.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 })}`;
    }
  };
  
  return (
    <nav className="bg-white dark:bg-gray-800 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <button 
                className="md:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={onMenuClick}
              >
                <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <Link to="/" className="flex items-center">
                <svg className="h-8 w-8 text-blue-600 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="ml-2 text-xl font-bold text-gray-800 dark:text-white">CryptexAI</span>
              </Link>
            </div>
            <div className="hidden md:ml-6 md:flex md:space-x-8">
              <Link 
                to="/" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 ${location.pathname === '/' ? 'border-blue-500 text-gray-800 dark:text-white' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-300 dark:hover:text-white'} text-sm font-medium`}
              >
                Dashboard
              </Link>
              <Link 
                to="/prediction" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 ${location.pathname === '/prediction' ? 'border-blue-500 text-gray-800 dark:text-white' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-300 dark:hover:text-white'} text-sm font-medium`}
              >
                Predictions
              </Link>
              <Link 
                to="/backtest" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 ${location.pathname === '/backtest' ? 'border-blue-500 text-gray-800 dark:text-white' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-300 dark:hover:text-white'} text-sm font-medium`}
              >
                Backtesting
              </Link>
              <Link 
                to="/model-comparison" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 ${location.pathname === '/model-comparison' ? 'border-blue-500 text-gray-800 dark:text-white' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-300 dark:hover:text-white'} text-sm font-medium`}
              >
                Model Comparison
              </Link>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex-shrink-0">
              <CryptoSelector />
            </div>
            <div className="flex-shrink-0">
              <div className="px-3 py-2 rounded-md bg-gray-100 dark:bg-gray-700">
                <div className="flex items-center">
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">{selectedCrypto.symbol}:</span>
                  {loading ? (
                    <span className="ml-1 text-sm font-bold animate-pulse">Loading...</span>
                  ) : (
                    <span className="ml-1 text-sm font-bold text-gray-800 dark:text-gray-200">{formatPrice(cryptoPrice)}</span>
                  )}
                  {!loading && cryptoPrice && (
                    <span className={`ml-2 text-xs font-medium ${priceChange.isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {priceChange.isPositive ? '↑' : '↓'} {priceChange.value}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <DarkModeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
} 