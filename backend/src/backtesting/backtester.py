import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

from ..models.base_model import BaseModel


class Backtester:
    """
    Backtester for simulating trading strategies on historical data.
    """
    
    def __init__(
        self,
        model: BaseModel,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission per trade
    ):
        """
        Initialize the backtester.
        
        Args:
            model: Model to use for predictions
            initial_balance: Initial account balance
            commission: Commission rate per trade
        """
        self.model = model
        self.initial_balance = initial_balance
        self.commission = commission
        
    def run(
        self,
        X_test: np.ndarray,
        prices: np.ndarray,
        dates: List[datetime],
        threshold: float = 0.01,  # 1% predicted change threshold for action
        position_size: float = 1.0,  # Fraction of available capital to use per trade
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            X_test: Test features for prediction
            prices: Actual prices corresponding to X_test
            dates: Dates corresponding to prices
            threshold: Minimum price change threshold to trigger a trade
            position_size: Fraction of available capital to use per trade
            
        Returns:
            Dictionary with backtest results
        """
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Initialize backtest variables
        balance = self.initial_balance
        btc_held = 0.0
        trades = []
        balances = [balance]
        positions = [0.0]  # BTC positions
        trade_dates = [dates[0]]
        
        # Run through the simulation
        for i in range(1, len(predictions)):
            prev_price = prices[i-1]
            current_price = prices[i]
            current_date = dates[i]
            
            # Calculate predicted percent change
            predicted_change = (predictions[i] - current_price) / current_price
            
            # Trading logic
            if btc_held == 0 and predicted_change > threshold:
                # Buy signal
                btc_to_buy = (balance * position_size) / current_price
                cost = btc_to_buy * current_price
                commission_fee = cost * self.commission
                
                if balance >= (cost + commission_fee):
                    balance -= (cost + commission_fee)
                    btc_held += btc_to_buy
                    
                    trades.append({
                        'date': current_date,
                        'type': 'BUY',
                        'price': current_price,
                        'amount': btc_to_buy,
                        'cost': cost + commission_fee,
                        'balance_after': balance
                    })
            
            elif btc_held > 0 and predicted_change < -threshold:
                # Sell signal
                sell_value = btc_held * current_price
                commission_fee = sell_value * self.commission
                balance += (sell_value - commission_fee)
                
                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'amount': btc_held,
                    'value': sell_value - commission_fee,
                    'balance_after': balance
                })
                
                btc_held = 0.0
            
            # Record state
            total_value = balance + (btc_held * current_price)
            balances.append(total_value)
            positions.append(btc_held)
            trade_dates.append(current_date)
        
        # Calculate metrics
        returns = (balances[-1] - self.initial_balance) / self.initial_balance
        
        # Daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(balances)):
            daily_return = (balances[i] - balances[i-1]) / balances[i-1]
            daily_returns.append(daily_return)
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = None
        if len(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        max_drawdown = 0
        peak = balances[0]
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Create the results dictionary
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': balances[-1],
            'returns': returns,
            'returns_pct': returns * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'trade_count': len(trades),
            'trades': trades,
            'balance_history': list(zip(trade_dates, balances)),
            'position_history': list(zip(trade_dates, positions)),
            'predictions': list(zip(dates, predictions)),
            'actual_prices': list(zip(dates, prices))
        }
        
        return results 