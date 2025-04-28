import React from 'react';

export default function StatCard({ 
  title, 
  value, 
  change, 
  icon,
  isLoading = false,
  isPositive = true,
  isMonetary = false,
  isPercentage = false,
  onClick = null
}) {
  const formatValue = (val) => {
    if (isLoading) return "Loading...";
    
    if (isMonetary) {
      return typeof val === 'number' 
        ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(val)
        : val;
    }
    
    if (isPercentage) {
      return typeof val === 'number' 
        ? `${val.toFixed(2)}%`
        : val;
    }
    
    return val;
  };

  const cardClasses = onClick 
    ? "transition-transform duration-200 transform hover:scale-105 hover:shadow-lg cursor-pointer"
    : "";
  
  return (
    <div 
      className={`bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md ${cardClasses}`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
          {isLoading ? (
            <div className="mt-1 h-6 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
          ) : (
            <p className="mt-1 text-xl font-semibold text-gray-800 dark:text-white">{formatValue(value)}</p>
          )}
        </div>
        <div className={`p-3 rounded-full ${isPositive ? 'bg-green-100 dark:bg-green-800/30' : 'bg-red-100 dark:bg-red-800/30'}`}>
          {icon || (isPositive ? (
            <svg className="h-6 w-6 text-green-600 dark:text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          ) : (
            <svg className="h-6 w-6 text-red-600 dark:text-red-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6" />
            </svg>
          ))}
        </div>
      </div>
      
      {change && (
        <div className="mt-4 flex items-center">
          {isLoading ? (
            <div className="h-4 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
          ) : (
            <div className={`flex items-center ${isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {isPositive ? (
                <svg className="h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              ) : (
                <svg className="h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              )}
              <span className="ml-1 text-sm font-medium">
                {change}
              </span>
            </div>
          )}
          <span className="ml-2 text-sm font-medium text-gray-500 dark:text-gray-400">from previous period</span>
        </div>
      )}
    </div>
  );
} 