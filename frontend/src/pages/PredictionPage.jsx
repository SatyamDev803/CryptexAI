import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import PriceChart from '../components/PriceChart';
import StatCard from '../components/StatCard';
import { getPredictions } from '../services/api';

export default function PredictionPage() {
  const { selectedCrypto, priceHistory } = useAppContext();
  
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [predictionDays, setPredictionDays] = useState(7);
  const [predictionResults, setPredictionResults] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  
  const modelOptions = [
    { value: 'lstm', label: 'LSTM', description: 'Long Short-Term Memory Neural Network' },
    { value: 'gru', label: 'GRU', description: 'Gated Recurrent Unit' },
    { value: 'transformer', label: 'Transformer', description: 'Attention-based model' }
  ];

  // Generate prediction with the selected model and days
  const generatePrediction = async () => {
    setIsGenerating(true);
    setError(null);
    
    try {
      console.log(`Generating predictions for ${selectedCrypto.id} using ${selectedModel} model for ${predictionDays} days`);
      
      // Call the API endpoint - this will fall back to mock data if the endpoint fails
      const result = await getPredictions(selectedModel, predictionDays, selectedCrypto.id);
      
      // Verify we have predictions data
      if (!result || !result.predictions || !Array.isArray(result.predictions) || result.predictions.length === 0) {
        throw new Error('Invalid prediction data received');
      }
      
      console.log(`Received ${result.predictions.length} prediction points`);
      
      // Create metrics object from the API response
      const metricsData = {
        hit_rate: result.metrics?.hit_rate ?? null,
        confidence: result.metrics?.hit_rate ? result.metrics.hit_rate / 100 : null,
      };
      
      setPredictionResults(result);
      setMetrics(metricsData);
      setIsGenerating(false);
    } catch (error) {
      console.error('Error generating prediction:', error);
      setError(`Failed to generate prediction: ${error.message}`);
      setIsGenerating(false);
    }
  };
  
  // Format date for table display
  const formatDate = (offset) => {
    const date = new Date();
    date.setDate(date.getDate() + offset);
    return date.toLocaleDateString();
  };
  
  // Clear prediction results
  const resetPredictions = () => {
    setPredictionResults(null);
    setMetrics(null);
  };
  
  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">{selectedCrypto.name} Price Prediction</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">Generate future price predictions based on deep learning models</p>
      
      {error && (
        <div className="mb-6 p-4 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md">
          <p>{error}</p>
          <p className="text-sm mt-2">Using mock data for visualization purposes.</p>
        </div>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold mb-4">Prediction Settings</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded-md bg-white dark:bg-gray-900 text-gray-800 dark:text-white"
              disabled={isGenerating}
            >
              {modelOptions.map(model => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {modelOptions.find(m => m.value === selectedModel)?.description}
            </p>
          </div>
          
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Prediction Days: {predictionDays}
            </label>
            <input
              type="range"
              min="1"
              max="30"
              value={predictionDays}
              onChange={(e) => setPredictionDays(parseInt(e.target.value))}
              className="w-full"
              disabled={isGenerating}
            />
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>1 day</span>
              <span>30 days</span>
            </div>
          </div>
          
          <button
            onClick={generatePrediction}
            disabled={isGenerating}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold rounded-md mb-4 flex justify-center items-center"
          >
            {isGenerating ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Generating...
              </>
            ) : "Generate Prediction"}
          </button>
          
          {predictionResults && (
            <button
              onClick={resetPredictions}
              className="w-full py-2 px-4 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 font-medium rounded-md"
            >
              Reset
            </button>
          )}
          
          {predictionResults && metrics && (
            <div className="mt-6 space-y-3">
              <h3 className="text-md font-medium text-gray-800 dark:text-gray-200">Prediction Metrics</h3>
              
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Hit Rate (±2%):</span>
                <span className="font-medium">{metrics.hit_rate}%</span>
              </div>
              
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Prediction Confidence:</span>
                <span className="font-medium">{(metrics.confidence * 100).toFixed(1)}%</span>
              </div>
              
              <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Est. Price Range:</span>
                <span className="font-medium">
                  {predictionResults.predictions[predictionDays - 1] && new Intl.NumberFormat('en-US', { 
                    style: 'currency', 
                    currency: 'USD',
                    maximumFractionDigits: 0
                  }).format(predictionResults.predictions[predictionDays - 1].price * 0.9)} - {predictionResults.predictions[predictionDays - 1] && new Intl.NumberFormat('en-US', { 
                    style: 'currency', 
                    currency: 'USD',
                    maximumFractionDigits: 0
                  }).format(predictionResults.predictions[predictionDays - 1].price * 1.1)}
                </span>
              </div>
            </div>
          )}
        </div>
        
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-6">
            <h2 className="text-lg font-semibold mb-4">Prediction Chart</h2>
            {!predictionResults ? (
              <div className="h-64 bg-gray-100 dark:bg-gray-700 rounded flex flex-col items-center justify-center">
                <svg className="w-12 h-12 text-gray-400 dark:text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                </svg>
                <p className="text-gray-500 dark:text-gray-400">Generate a prediction to see the forecast chart</p>
              </div>
            ) : (
              <PriceChart
                data={priceHistory}
                predictions={predictionResults.predictions}
                showPrediction={true}
                height={320}
                title={`${selectedCrypto.name} ${predictionDays}-Day Price Prediction`}
              />
            )}
          </div>
          
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Predicted Values</h2>
            {!predictionResults ? (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                No prediction data available yet. Generate a prediction to see results.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Date</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Predicted Price</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Confidence Interval</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
                    {predictionResults.predictions.map((prediction, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {formatDate(index + 1)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800 dark:text-white">
                          {new Intl.NumberFormat('en-US', { 
                            style: 'currency', 
                            currency: 'USD' 
                          }).format(prediction.price)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                          {prediction.lower_bound && prediction.upper_bound ? 
                            `${new Intl.NumberFormat('en-US', { 
                              style: 'currency', 
                              currency: 'USD',
                              maximumFractionDigits: 0 
                            }).format(prediction.lower_bound)} - ${new Intl.NumberFormat('en-US', { 
                              style: 'currency', 
                              currency: 'USD',
                              maximumFractionDigits: 0
                            }).format(prediction.upper_bound)}` :
                            `±${(prediction.price * 0.1).toFixed(0)}`
                          }
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 