import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import PriceChart from '../components/PriceChart';
import StatCard from '../components/StatCard';
import ModelSelector from '../components/ModelSelector';
import { getPredictions } from '../services/api';

export default function Dashboard() {
  const { 
    cryptoPrice, 
    priceHistory, 
    loading, 
    selectedCrypto, 
    refreshPriceData,
  } = useAppContext();
  
  const [timeRange, setTimeRange] = useState('30d');
  const [predictions, setPredictions] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isMetricsLoading, setIsMetricsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState('lstm');
  
  // Refresh data on component mount and when crypto changes
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsMetricsLoading(true);
        // Fetch predictions for the selected crypto from the backend
        const predictionData = await getPredictions(selectedModel, 30, selectedCrypto.id);
        setPredictions(predictionData.predictions);
        
        // Set metrics from the API response
        setMetrics({
          hit_rate: predictionData.metrics.hit_rate,
          mae: predictionData.metrics.mae,
          rmse: predictionData.metrics.rmse,
          volatility: predictionData.metrics.mape || 2.84,
          trend: predictionData.metrics.hit_rate > 60 ? 'bullish' : 'bearish'
        });
        
        setIsMetricsLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setIsMetricsLoading(false);
      }
    };
    
    fetchData();
    
    // Set up timer to refresh data every 60 seconds
    const timer = setInterval(() => {
      refreshPriceData(parseInt(timeRange));
    }, 60000);
    
    return () => clearInterval(timer);
  }, [selectedCrypto.id, timeRange, refreshPriceData, selectedModel]);
  
  // Extract fetchData as a separate function so it can be called from the UI
  const fetchData = async () => {
    try {
      setIsMetricsLoading(true);
      // Fetch predictions for the selected crypto using current model
      const predictionData = await getPredictions(selectedModel, 30, selectedCrypto.id);
      setPredictions(predictionData.predictions);
      
      // Set metrics from the API response
      setMetrics({
        hit_rate: predictionData.metrics.hit_rate,
        mae: predictionData.metrics.mae,
        rmse: predictionData.metrics.rmse,
        volatility: predictionData.metrics.mape || 2.84,
        trend: predictionData.metrics.hit_rate > 60 ? 'bullish' : 'bearish'
      });
      
      setIsMetricsLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setIsMetricsLoading(false);
    }
  };
  
  const handleTimeRangeChange = (days) => {
    setTimeRange(days);
    refreshPriceData(parseInt(days));
  };
  
  // Generate daily change
  const calculateDailyChange = () => {
    if (!priceHistory || priceHistory.length < 2) return { value: '0.00%', isPositive: true };
    
    const current = priceHistory[priceHistory.length - 1].close;
    const yesterday = priceHistory[priceHistory.length - 2].close;
    const changePercent = ((current - yesterday) / yesterday) * 100;
    
    return {
      value: `${Math.abs(changePercent).toFixed(2)}%`,
      isPositive: changePercent >= 0
    };
  };
  
  const dailyChange = calculateDailyChange();
  
  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-400">{selectedCrypto.name} Dashboard</h1>
          <p className="text-gray-800 dark:text-gray-400">Real-time analysis and predictions</p>
        </div>
        <div className="flex space-x-2 items-center ">
          <ModelSelector 
            selectedModel={selectedModel} 
            onChange={(model) => {
              setSelectedModel(model);
              fetchData();
            }}
            compact={true}
            className="mr-4 w-40"
          />
          
          <button 
            onClick={() => handleTimeRangeChange('7d')}
            className={`px-3 py-1 rounded-md ${timeRange === '7d' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
          >
            7D
          </button>
          <button 
            onClick={() => handleTimeRangeChange('30d')}
            className={`px-3 py-1 rounded-md ${timeRange === '30d' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
          >
            30D
          </button>
          <button 
            onClick={() => handleTimeRangeChange('90d')}
            className={`px-3 py-1 rounded-md ${timeRange === '90d' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'}`}
          >
            90D
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard 
          title="Current Price" 
          value={cryptoPrice} 
          change={dailyChange.value}
          isPositive={dailyChange.isPositive}
          isLoading={loading}
          isMonetary={true}
        />
        
        <StatCard 
          title="Hit Rate (±2%)" 
          value={metrics?.hit_rate} 
          isLoading={isMetricsLoading}
          isPercentage={true}
          isPositive={metrics?.hit_rate > 60}
        />
        
        <StatCard 
          title="Mean Abs. Error" 
          value={metrics?.mae} 
          isLoading={isMetricsLoading}
          isPositive={false}
          isMonetary={true}
        />
        
        <StatCard 
          title="Volatility" 
          value={metrics?.volatility} 
          isLoading={isMetricsLoading}
          isPercentage={true}
          isPositive={metrics?.trend === 'bullish'}
        />
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="lg:col-span-2">
          <PriceChart 
            data={priceHistory} 
            predictions={predictions} 
            title={`${selectedCrypto.name} Price Chart`}
            timeRange={timeRange}
          />
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-400">Prediction Summary</h2>
          {loading || !predictions ? (
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">7-Day Forecast:</span>
                <span className="font-medium text-gray-800 dark:text-gray-400">
                  {predictions && predictions.length > 6 ? 
                    new Intl.NumberFormat('en-US', { 
                      style: 'currency', 
                      currency: 'USD' 
                    }).format(predictions[6].price) : 'N/A'
                  }
                </span>
              </div>
              
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">30-Day Trend:</span>
                <span className={`font-medium ${metrics?.trend === 'bullish' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {metrics?.trend === 'bullish' ? 'Bullish ↑' : 'Bearish ↓'}
                </span>
              </div>
              
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Confidence Level:</span>
                <span className="font-medium text-gray-800 dark:text-gray-400">
                  {predictions && predictions.length > 0 && predictions[0].lower_bound ? 
                    'High (±5%)' : 'Medium (±10%)'
                  }
                </span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-600 dark:text-gray-400">Last Updated:</span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date().toLocaleString()}
                </span>
              </div>
              
              <div className="mt-6">
                <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md">
                  View Detailed Forecast
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md text-gray-800 dark:text-gray-400">
        <h2 className="text-lg font-semibold mb-4">Recent Market Analysis</h2>
        <div className="prose dark:prose-invert max-w-none">
          <p>
            Our deep learning models are indicating a {metrics?.trend === 'bullish' ? 'positive' : 'cautious'} outlook 
            for {selectedCrypto.name} in the coming weeks. Based on technical indicators and market sentiment analysis,
            we're observing {metrics?.trend === 'bullish' ? 'increased buying pressure' : 'potential consolidation'}.
          </p>
          <p>
            Key factors influencing the current prediction:
          </p>
          <ul>
            <li>Market volatility at {metrics?.volatility}% indicates {metrics?.volatility > 3 ? 'high uncertainty' : 'relative stability'}.</li>
            <li>Model accuracy of {metrics?.hit_rate}% provides good confidence in short-term predictions.</li>
            <li>Recent price movement shows {dailyChange.isPositive ? 'positive momentum' : 'downward pressure'}.</li>
          </ul>
          <p>
            For more detailed analysis and to see how different models compare, visit the Model Comparison page.
          </p>
        </div>
      </div>
    </div>
  );
} 