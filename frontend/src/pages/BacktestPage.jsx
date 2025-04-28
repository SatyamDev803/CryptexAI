import React, { useState, useRef, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { runBacktest } from '../services/api';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

export default function BacktestPage() {
  const { selectedCrypto } = useAppContext();
  const equityRef = useRef(null);
  const monthlyRef = useRef(null);
  const drawdownRef = useRef(null);
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [backtestPeriod, setBacktestPeriod] = useState('1y');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [backtestResults, setBacktestResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Store chart instances to destroy them before creating new ones
  const [equityChart, setEquityChart] = useState(null);
  const [monthlyChart, setMonthlyChart] = useState(null);
  const [drawdownChart, setDrawdownChart] = useState(null);

  // Map period strings to days
  const periodToDays = {
    '1m': 30,
    '3m': 90,
    '6m': 180,
    '1y': 365,
    '3y': 1095,
    'all': 1825 // ~5 years
  };

  // Destroy all charts to prevent "Canvas already in use" errors
  const destroyCharts = () => {
    if (equityChart) {
      equityChart.destroy();
      setEquityChart(null);
    }
    if (monthlyChart) {
      monthlyChart.destroy();
      setMonthlyChart(null);
    }
    if (drawdownChart) {
      drawdownChart.destroy();
      setDrawdownChart(null);
    }
  };

  const handleRunBacktest = async () => {
    setIsLoading(true);
    setError(null);
    
    // Destroy any existing charts
    destroyCharts();

    try {
      const days = periodToDays[backtestPeriod];
      console.log(`Running backtest for ${selectedCrypto.id} using ${selectedModel} model for ${days} days with initial capital $${initialCapital}`);

      // Run the backtest with real data 
      const results = await runBacktest(
        selectedModel, 
        days, 
        0.01, // Default threshold
        initialCapital,
        selectedCrypto.id
      );

      // Verify we have valid data
      if (!results || !results.trades || !results.balance_history) {
        throw new Error('Invalid or incomplete backtest data received');
      }

      console.log(`Backtest completed with ${results.trades_count} trades`);
      setBacktestResults(results);
      setIsLoading(false);
    } catch (error) {
      console.error('Error running backtest:', error);
      setError(`Failed to run backtest with real data: ${error.message}. Please try again or contact support if the issue persists.`);
      setBacktestResults(null); // Clear any previous results
      setIsLoading(false);
    }
  };

  // Cleanup charts on component unmount
  useEffect(() => {
    return () => {
      destroyCharts();
    };
  }, [destroyCharts]);

  useEffect(() => {
    if (backtestResults) {
      // Make sure any old charts are destroyed
      destroyCharts();
      
      // Only create charts if we have valid data
      if (backtestResults.balance_history && backtestResults.balance_history.length > 0) {
        try {
          // Equity Curve chart
          const eqCtx = equityRef.current.getContext('2d');
          const newEquityChart = new Chart(eqCtx, {
            type: 'line',
            data: {
              labels: backtestResults.balance_history.map(([d]) => d),
              datasets: [{
                label: 'Equity Curve',
                data: backtestResults.balance_history.map(([, b]) => b),
                borderColor: 'rgba(75,192,192,1)',
                fill: false
              }]
            },
            options: { responsive: true, maintainAspectRatio: false }
          });
          setEquityChart(newEquityChart);
          
          // Monthly Returns chart
          const monthlyMap = {};
          backtestResults.balance_history.forEach(([d, b]) => {
            const m = d.slice(0,7);
            if (!monthlyMap[m]) monthlyMap[m] = { first: b, last: b };
            else monthlyMap[m].last = b;
          });
          const months = Object.keys(monthlyMap);
          const returnsData = months.map(m => ((monthlyMap[m].last - monthlyMap[m].first) / monthlyMap[m].first * 100).toFixed(2));
          const monCtx = monthlyRef.current.getContext('2d');
          const newMonthlyChart = new Chart(monCtx, {
            type: 'bar',
            data: {
              labels: months,
              datasets: [{ label: 'Monthly Returns %', data: returnsData, backgroundColor: 'rgba(153,102,255,0.6)' }]
            },
            options: { responsive: true, maintainAspectRatio: false }
          });
          setMonthlyChart(newMonthlyChart);
          
          // Drawdown chart
          let peak = -Infinity;
          const dd = backtestResults.balance_history.map(([, b]) => {
            if (b > peak) peak = b;
            return ((peak - b) / peak * 100).toFixed(2);
          });
          const ddCtx = drawdownRef.current.getContext('2d');
          const newDrawdownChart = new Chart(ddCtx, {
            type: 'line',
            data: {
              labels: backtestResults.balance_history.map(([d]) => d),
              datasets: [{ label: 'Drawdown %', data: dd, borderColor: 'rgba(255,99,132,1)', fill: false }]
            },
            options: { responsive: true, maintainAspectRatio: false }
          });
          setDrawdownChart(newDrawdownChart);
        } catch (err) {
          console.error("Error creating charts:", err);
          setError(`Error creating charts: ${err.message}`);
        }
      } else {
        console.warn("No balance history data available for charting");
        setError("No balance history data available for charting");
      }
    }
  }, [backtestResults, destroyCharts]);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-400">Backtesting</h1>
      <p className="mb-8 text-gray-800 dark:text-gray-400">Test trading strategies based on model predictions using historical data.</p>

      {error && (
        <div className="mb-6 p-4 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md">
          <p>{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-6">
            <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-400">Backtest Settings</h2>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="lstm">LSTM Model</option>
                <option value="gru">GRU Model</option>
                <option value="transformer">Transformer Model</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Backtest Period
              </label>
              <select
                value={backtestPeriod}
                onChange={(e) => setBacktestPeriod(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              >
                <option value="1m">1 Month</option>
                <option value="3m">3 Months</option>
                <option value="6m">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="3y">3 Years</option>
                <option value="all">All Data</option>
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Initial Capital ($)
              </label>
              <input
                type="number"
                min="100"
                step="100"
                value={initialCapital}
                onChange={(e) => setInitialCapital(parseFloat(e.target.value))}
                className="w-full p-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              />
            </div>

            <button 
              onClick={handleRunBacktest}
              disabled={isLoading}
              className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold rounded-md flex justify-center items-center"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Running...
                </>
              ) : "Run Backtest"}
            </button>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-400">Backtest Results</h2>
            {!backtestResults ? (
              <div className="text-center py-4 text-gray-500 dark:text-gray-400">
                Run a backtest to see results
              </div>
            ) : (
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Initial Capital</p>
                  <p className="text-lg font-medium">
                    ${backtestResults.initial_balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Final Capital</p>
                  <p className="text-lg font-medium text-green-600 dark:text-green-400">
                    ${backtestResults.final_balance.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Total Return</p>
                  <p className={`text-lg font-medium ${backtestResults.returns >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {backtestResults.returns >= 0 ? '+' : ''}{backtestResults.returns.toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</p>
                  <p className="text-lg font-medium">{backtestResults.sharpe_ratio.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</p>
                  <p className="text-lg font-medium text-red-600 dark:text-red-400">-{backtestResults.max_drawdown.toFixed(2)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Number of Trades</p>
                  <p className="text-lg font-medium">{backtestResults.trades_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Win Rate</p>
                  <p className="text-lg font-medium">{backtestResults.win_rate.toFixed(2)}%</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {backtestResults && (
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
            <div className="h-64">
              <canvas ref={equityRef} className="w-full h-full" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Monthly Returns</h2>
            <div className="h-64">
              <canvas ref={monthlyRef} className="w-full h-full" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Drawdown</h2>
            <div className="h-64">
              <canvas ref={drawdownRef} className="w-full h-full" />
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-4">Trade History</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th>Date</th>
                    <th>Type</th>
                    <th>Price</th>
                    <th>Quantity</th>
                    <th>Balance After</th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {backtestResults.trades.map((trade, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'bg-white dark:bg-gray-900' : 'bg-gray-50 dark:bg-gray-800'}>
                      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">{trade.date}</td>
                      <td className={`px-6 py-4 text-sm ${trade.type === 'BUY' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>{trade.type}</td>
                      <td className="px-6 py-4 text-sm text-gray-800 dark:text-white">${trade.price.toLocaleString()}</td>
                      <td className="px-6 py-4 text-sm text-gray-800 dark:text-white">
                        {typeof trade.amount === 'number'
                          ? trade.amount.toFixed(4)
                          : trade.amount}
                        {' '}{selectedCrypto.symbol}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                        ${trade.balance_after != null
                          ? trade.balance_after.toFixed(2)
                          : (trade.value != null && typeof trade.value === 'number'
                              ? trade.value.toFixed(2)
                              : trade.value)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}