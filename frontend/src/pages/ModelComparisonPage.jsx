import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import PriceChart from '../components/PriceChart';
import { modelComparison } from '../services/api';

export default function ModelComparisonPage() {
  const { selectedCrypto } = useAppContext();
  
  const [predictionDays, setPredictionDays] = useState(14);
  const [comparisonData, setComparisonData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch comparison data for all models
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        console.log(`Fetching model comparison data for ${selectedCrypto.id} for ${predictionDays} days`);
        
        // Call real API endpoint for model comparison (will fall back to mock if needed)
        const result = await modelComparison(selectedCrypto.id, predictionDays);
        
        // Verify we have valid data
        if (!result || !result.predictions || !result.metrics) {
          throw new Error('Invalid model comparison data received');
        }
        
        setComparisonData(result);
        setMetrics(result.metrics);
        console.log('Successfully loaded model comparison data');
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching comparison data:', error);
        setError(`Failed to fetch model comparison data: ${error.message}`);
        // Note: The API service will have already fallen back to mock data
        setIsLoading(false);
      }
    };
    
    fetchData();
  }, [selectedCrypto.id, predictionDays]);
  
  return (
    <div>
      <h1 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-400">Model Comparison</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">Compare the performance of different prediction models</p>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md">
          <p>{error}</p>
          <p className="text-sm mt-2 text-gray-600 dark:text-gray-400">Using mock data for visualization purposes.</p>
        </div>
      )}
      
      <div className="mb-6 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-400">Prediction Period</h2>
            <p className="text-gray-600 dark:text-gray-400">Adjust the time range to see how different models perform over time.</p>
          </div>
          <div className="w-64">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Days to Predict: {predictionDays}
            </label>
            <input
              type="range"
              min="7"
              max="30"
              value={predictionDays}
              onChange={(e) => setPredictionDays(parseInt(e.target.value))}
              className="w-full"
              disabled={isLoading}
            />
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>7 days</span>
              <span>30 days</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Prediction Comparison</h2>
            {isLoading ? (
              <div className="h-80 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
                <div className="text-gray-500 dark:text-gray-400">
                  <svg className="animate-spin h-8 w-8 mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <p>Loading comparison data...</p>
                </div>
              </div>
            ) : (
              <PriceChart
                comparisonData={comparisonData}
                height={'h-80'}
                title={`${selectedCrypto.name} ${predictionDays}-Day Prediction Comparison`}
              />
            )}
          </div>
        </div>
        
        <div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Model Performance</h2>
            {isLoading ? (
              <div className="space-y-4">
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
                <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
              </div>
            ) : (
              <div className="space-y-6">
                <div>
                  <h3 className="text-md font-medium text-blue-600 dark:text-blue-400 mb-2">LSTM Model</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Hit Rate:</span>
                      <span className="font-medium">{metrics?.lstm.hit_rate}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Error:</span>
                      <span className="font-medium">
                        {new Intl.NumberFormat('en-US', { 
                          style: 'currency', 
                          currency: 'USD' 
                        }).format(metrics?.lstm.mae)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Percentage Error:</span>
                      <span className="font-medium">{metrics?.lstm.mape.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-md font-medium text-red-600 dark:text-red-400 mb-2">GRU Model</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Hit Rate:</span>
                      <span className="font-medium">{metrics?.gru.hit_rate}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Error:</span>
                      <span className="font-medium">
                        {new Intl.NumberFormat('en-US', { 
                          style: 'currency', 
                          currency: 'USD' 
                        }).format(metrics?.gru.mae)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Percentage Error:</span>
                      <span className="font-medium">{metrics?.gru.mape.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-md font-medium text-green-600 dark:text-green-400 mb-2">Transformer Model</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Hit Rate:</span>
                      <span className="font-medium">{metrics?.transformer.hit_rate}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Error:</span>
                      <span className="font-medium">
                        {new Intl.NumberFormat('en-US', { 
                          style: 'currency', 
                          currency: 'USD' 
                        }).format(metrics?.transformer.mae)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">Mean Absolute Percentage Error:</span>
                      <span className="font-medium">{metrics?.transformer.mape.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h2 className="text-lg font-semibold mb-4">Detailed Comparison</h2>
        {isLoading ? (
          <div className="animate-pulse space-y-4">
            <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
            <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Metric</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">LSTM</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">GRU</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Transformer</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">Hit Rate</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.lstm.hit_rate}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.gru.hit_rate}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.transformer.hit_rate}%</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">Mean Absolute Error</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.lstm.mae)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.gru.mae)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.transformer.mae)}
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">Root Mean Squared Error</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.lstm.rmse)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.gru.rmse)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(metrics?.transformer.rmse)}
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">Mean Absolute Percentage Error</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.lstm.mape.toFixed(1)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.gru.mape.toFixed(1)}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metrics?.transformer.mape.toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
} 