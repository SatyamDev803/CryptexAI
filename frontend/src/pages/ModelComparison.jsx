import React, { useState } from 'react';

export default function ModelComparison() {
  const [timeframe, setTimeframe] = useState('1m');
  const [models, setModels] = useState(['lstm', 'gru', 'transformer']);
  
  const metrics = [
    { name: 'Mean Absolute Error (MAE)', lstm: 325.45, gru: 342.18, transformer: 298.76 },
    { name: 'Root Mean Square Error (RMSE)', lstm: 428.91, gru: 456.32, transformer: 389.45 },
    { name: 'Mean Absolute Percentage Error (MAPE)', lstm: 1.24, gru: 1.38, transformer: 1.15 },
    { name: 'Directional Accuracy', lstm: 68.5, gru: 65.2, transformer: 72.1 },
    { name: 'Sharpe Ratio', lstm: 1.45, gru: 1.32, transformer: 1.63 }
  ];
  
  const toggleModel = (model) => {
    if (models.includes(model)) {
      setModels(models.filter(m => m !== model));
    } else {
      setModels([...models, model]);
    }
  };
  
  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Model Comparison</h1>
      <p className="mb-8">Compare performance metrics between different predictive models.</p>
      
      <div className="mb-8 bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timeframe
            </label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="p-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="1w">Last Week</option>
              <option value="1m">Last Month</option>
              <option value="3m">Last 3 Months</option>
              <option value="6m">Last 6 Months</option>
              <option value="1y">Last Year</option>
              <option value="all">All Time</option>
            </select>
          </div>
          
          <div className="flex-grow">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Models to Compare
            </label>
            <div className="flex gap-4">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="lstm"
                  checked={models.includes('lstm')}
                  onChange={() => toggleModel('lstm')}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="lstm" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  LSTM
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="gru"
                  checked={models.includes('gru')}
                  onChange={() => toggleModel('gru')}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="gru" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  GRU
                </label>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="transformer"
                  checked={models.includes('transformer')}
                  onChange={() => toggleModel('transformer')}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <label htmlFor="transformer" className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                  Transformer
                </label>
              </div>
            </div>
          </div>
          
          <div>
            <button className="py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-md">
              Update Comparison
            </button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4">Performance Metrics</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Metric</th>
                  {models.includes('lstm') && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">LSTM</th>
                  )}
                  {models.includes('gru') && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">GRU</th>
                  )}
                  {models.includes('transformer') && (
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Transformer</th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {metrics.map((metric, i) => (
                  <tr key={i}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">{metric.name}</td>
                    {models.includes('lstm') && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metric.lstm}</td>
                    )}
                    {models.includes('gru') && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metric.gru}</td>
                    )}
                    {models.includes('transformer') && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{metric.transformer}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4">Metrics Comparison Chart</h2>
          <div className="h-80 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
            <p>Radar chart comparing model metrics will be displayed here</p>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4">Price Prediction Comparison</h2>
          <div className="h-80 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
            <p>Line chart comparing model predictions will be displayed here</p>
          </div>
        </div>
        
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4">Prediction Error Distribution</h2>
          <div className="h-80 bg-gray-100 dark:bg-gray-700 rounded flex items-center justify-center">
            <p>Error distribution histogram will be displayed here</p>
          </div>
        </div>
      </div>
    </div>
  );
} 