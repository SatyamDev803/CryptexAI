import { useEffect, useRef, useState } from 'react';
import { Chart, registerables } from 'chart.js';
import { format } from 'date-fns';
import { useAppContext } from '../context/AppContext';

// Register all Chart.js components
Chart.register(...registerables);

export default function PriceChart({ 
  data, 
  predictions, 
  title, 
  timeRange = '1M',
  showLegend = true,
  height = "h-80", 
  showPrediction = true,
  comparisonData = null // For model comparison view
}) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const { selectedCrypto } = useAppContext();
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (data) {
      console.log(`Using real price data for ${selectedCrypto.id}, ${data.length} data points`);
      setChartData(data);
    }
  }, [data, selectedCrypto.id]);

  useEffect(() => {
    // Clean up previous chart if it exists
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    if (!chartRef.current || (!chartData && !comparisonData)) return;

    const ctx = chartRef.current.getContext('2d');
    
    let datasets = [];
    
    // Setup for regular price chart with predictions
    if (chartData) {
      const dates = chartData.map(item => format(new Date(item.date), 'MMM dd'));
      const prices = chartData.map(item => item.close);
      
      // For debugging price data
      console.log(`Chart price range: ${Math.min(...prices)} - ${Math.max(...prices)}`);
      
      let predictionDates = [];
      let predictionPrices = [];
      let lowerBounds = [];
      let upperBounds = [];
      
      if (predictions && predictions.length > 0 && showPrediction) {
        predictionDates = predictions.map(item => format(new Date(item.date), 'MMM dd'));
        predictionPrices = predictions.map(item => item.price);
        
        // For debugging prediction data
        console.log(`Prediction price range: ${Math.min(...predictionPrices)} - ${Math.max(...predictionPrices)}`);
        
        if (predictions[0].lower_bound) {
          lowerBounds = predictions.map(item => item.lower_bound);
          upperBounds = predictions.map(item => item.upper_bound);
        }
      }
      
      datasets = [
        {
          label: 'Actual Price',
          data: [...prices, ...Array(predictionDates.length).fill(null)],
          borderColor: 'rgb(59, 130, 246)', // Blue
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          fill: false,
          tension: 0.4,
        }
      ];
      
      if (showPrediction && predictions && predictions.length > 0) {
        datasets.push({
          label: 'Predicted Price',
          data: [...Array(dates.length).fill(null), ...predictionPrices],
          borderColor: 'rgb(239, 68, 68)', // Red
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
        });
        
        // Add confidence interval if available
        if (lowerBounds.length > 0 && upperBounds.length > 0) {
          datasets.push({
            label: 'Confidence Interval',
            data: [...Array(dates.length).fill(null), ...Array(predictionDates.length).fill(null)],
            borderColor: 'rgba(239, 68, 68, 0)', // Transparent
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            fill: {
              target: '+1',
              above: 'rgba(239, 68, 68, 0.1)',
              below: 'rgba(239, 68, 68, 0.1)'
            },
            pointRadius: 0,
            tension: 0.4
          });
        }
      }
      
      // Creating the chart
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [...dates, ...(showPrediction ? predictionDates : [])],
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: !!title,
              text: title || `${selectedCrypto.name} Price Chart`,
              font: {
                size: 16,
                weight: 'bold'
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) {
                    label += ': ';
                  }
                  if (context.parsed.y !== null) {
                    label += new Intl.NumberFormat('en-US', { 
                      style: 'currency', 
                      currency: 'USD',
                      minimumFractionDigits: 2
                    }).format(context.parsed.y);
                  }
                  return label;
                }
              }
            },
            legend: {
              display: showLegend,
              position: 'top',
            }
          },
          scales: {
            x: {
              grid: {
                display: false
              },
              ticks: {
                maxTicksLimit: 10
              }
            },
            y: {
              grid: {
                color: 'rgba(156, 163, 175, 0.1)'
              },
              ticks: {
                callback: function(value) {
                  return new Intl.NumberFormat('en-US', { 
                    style: 'currency', 
                    currency: 'USD',
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                  }).format(value);
                }
              }
            }
          },
          interaction: {
            intersect: false,
            mode: 'index'
          },
          elements: {
            point: {
              radius: 0 // Hide points by default
            }
          },
          hover: {
            mode: 'index',
            intersect: false
          }
        }
      });
    }
    // Setup for model comparison chart
    else if (comparisonData) {
      const { dates, lstm, gru, transformer } = comparisonData.predictions;
      
      datasets = [];
      
      if (lstm) {
        datasets.push({
          label: 'LSTM',
          data: lstm,
          borderColor: 'rgb(59, 130, 246)', // Blue
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          fill: false,
          tension: 0.4,
        });
      }
      
      if (gru) {
        datasets.push({
          label: 'GRU',
          data: gru,
          borderColor: 'rgb(239, 68, 68)', // Red
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          fill: false,
          tension: 0.4,
        });
      }
      
      if (transformer) {
        datasets.push({
          label: 'Transformer',
          data: transformer,
          borderColor: 'rgb(16, 185, 129)', // Green
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          borderWidth: 2,
          pointRadius: 1,
          fill: false,
          tension: 0.4,
        });
      }
      
      // Creating the comparison chart
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: dates,
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: !!title,
              text: title || `${selectedCrypto.name} Price Prediction Comparison`,
              font: {
                size: 16,
                weight: 'bold'
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) {
                    label += ': ';
                  }
                  if (context.parsed.y !== null) {
                    label += new Intl.NumberFormat('en-US', { 
                      style: 'currency', 
                      currency: 'USD',
                      minimumFractionDigits: 2
                    }).format(context.parsed.y);
                  }
                  return label;
                }
              }
            },
            legend: {
              display: showLegend,
              position: 'top',
            }
          },
          scales: {
            x: {
              grid: {
                display: false
              },
              ticks: {
                maxTicksLimit: 10
              }
            },
            y: {
              grid: {
                color: 'rgba(156, 163, 175, 0.1)'
              },
              ticks: {
                callback: function(value) {
                  return new Intl.NumberFormat('en-US', { 
                    style: 'currency', 
                    currency: 'USD',
                    minimumFractionDigits: 0,
                    maximumFractionDigits: 0
                  }).format(value);
                }
              }
            }
          },
          interaction: {
            intersect: false,
            mode: 'index'
          },
          elements: {
            point: {
              radius: 0 // Hide points by default
            }
          },
          hover: {
            mode: 'index',
            intersect: false
          }
        }
      });
    }
    
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [chartData, predictions, title, timeRange, selectedCrypto.name, showPrediction, showLegend, comparisonData]);

  return (
    <div className={`bg-white dark:bg-gray-800 p-4 rounded-lg shadow-md ${height}`}>
      {!chartData && !comparisonData ? (
        <div className="h-full flex items-center justify-center">
          <div className="text-gray-500 dark:text-gray-400">
            <svg className="animate-spin h-8 w-8 mx-auto mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p>Loading chart data...</p>
          </div>
        </div>
      ) : (
        <canvas ref={chartRef}></canvas>
      )}
    </div>
  );
} 