import React from 'react';

export default function ModelSelector({ 
  selectedModel,
  onChange,
  className,
  label = "Select Model",
  compact = false,
  models = [
    { id: 'lstm', name: 'LSTM Model', description: 'Long Short-Term Memory neural network' },
    { id: 'gru', name: 'GRU Model', description: 'Gated Recurrent Unit neural network' },
    { id: 'transformer', name: 'Transformer', description: 'Transformer-based neural network' }
  ]
}) {
  return (
    <div className={`${className || ''}`}>
      {!compact && (
        <label htmlFor="model-selector" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          {label}
        </label>
      )}
      <select
        id="model-selector"
        value={selectedModel}
        onChange={(e) => onChange(e.target.value)}
        className={`${compact ? 'text-sm' : 'mt-1'} block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white`}
      >
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
      {!compact && (
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          {models.find(m => m.id === selectedModel)?.description || 'Select a model to make predictions'}
        </p>
      )}
    </div>
  );
} 