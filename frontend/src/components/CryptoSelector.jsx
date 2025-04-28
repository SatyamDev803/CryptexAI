import React from 'react';
import { useAppContext } from '../context/AppContext';

export default function CryptoSelector() {
  const { cryptocurrencies, selectedCrypto, changeCrypto } = useAppContext();

  return (
    <div className="relative">
      <select
        className="block w-full px-4 py-2 pr-8 leading-tight bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 text-gray-700 dark:text-gray-200"
        value={selectedCrypto.id}
        onChange={(e) => changeCrypto(e.target.value)}
      >
        {cryptocurrencies.map((crypto) => (
          <option key={crypto.id} value={crypto.id}>
            {crypto.symbol} - {crypto.name}
          </option>
        ))}
      </select>
      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 dark:text-gray-300">
        <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
          <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
        </svg>
      </div>
    </div>
  );
} 