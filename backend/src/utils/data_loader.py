import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List, Any, Union, Optional


class DataLoader:
    """Utility for loading and preprocessing cryptocurrency data."""
    
    def __init__(self, symbol: str = "BTC-USD"):
        """
        Initialize the data loader.
        
        Args:
            symbol: Symbol to fetch data for
        """
        self.symbol = symbol
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feat_scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        
    def fetch_data(
        self,
        start_date: Union[str, datetime] = "2014-09-17",
        end_date: Union[str, datetime] = None,
        source: str = "yfinance",
        csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency data.
        
        Args:
            start_date: Start date for fetching data
            end_date: End date for fetching data (default: today)
            source: Data source ("yfinance" or "csv")
            csv_path: Path to CSV file if source is "csv"
            
        Returns:
            DataFrame containing price data
        """
        if end_date is None:
            end_date = datetime.now().date().isoformat()

        # Try to get data from yfinance first
        if source == "yfinance":
            try:
                data = yf.download(
                    self.symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=False,
                )
                if data.empty:
                    print(f"Warning: yfinance returned no data for {self.symbol}. Using synthetic data.")
                    return self._generate_synthetic_data(start_date, end_date)
                self.data = data
                return data
            except Exception as e:
                print(f"Warning: yfinance fetch error: {e}. Using synthetic data.")
                return self._generate_synthetic_data(start_date, end_date)
        
        # Try to load from CSV if available
        if source == "csv" and csv_path:
            try:
                data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                self.data = data
                return data
            except Exception as e:
                raise RuntimeError(f"Error loading data from CSV: {e}")
        
        # No source matched or CSV not provided: fallback synthetic
        print("Warning: No valid data source. Using synthetic data.")
        return self._generate_synthetic_data(start_date, end_date)
        
    def _generate_synthetic_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate synthetic price data when no real data is available.
        
        Args:
            start_date: Start date 
            end_date: End date
            
        Returns:
            DataFrame with synthetic price data
        """
        # Convert dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Set base price based on crypto
        if 'BTC' in self.symbol:
            base_price = 84000.0
        elif 'ETH' in self.symbol:
            base_price = 3000.0
        elif 'SOL' in self.symbol:
            base_price = 150.0
        elif 'ADA' in self.symbol:
            base_price = 0.45
        elif 'XRP' in self.symbol:
            base_price = 0.50
        else:
            base_price = 100.0
            
        # Generate random price movements
        np.random.seed(42)  # for reproducibility
        price_changes = np.random.normal(0, 0.02, len(date_range))  # 2% daily volatility
        prices = [base_price]
        
        # Accumulate price changes
        for change in price_changes[:-1]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
        # Create DataFrame
        data = pd.DataFrame({
            'Open': [p * 0.99 for p in prices],
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(date_range))
        }, index=date_range)
        
        self.data = data
        return data
    
    def prepare_data(
        self,
        prediction_days: int = 80,
        future_day: int = 1,
        target_column: str = "Close",
        test_split_date: Optional[Union[str, datetime]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Prepare data for training and testing.
        
        Args:
            prediction_days: Number of days to use for prediction
            future_day: Number of days in the future to predict
            target_column: Column to predict
            test_split_date: Date to split training and testing data
            test_size: Fraction of data to use for testing if test_split_date is None
            
        Returns:
            Dictionary containing training and testing data
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        data = self.data.copy()
        
        # Feature engineering: momentum, SMAs, EMAs, RSI, Bollinger Bands, returns, volume ratio
        data['ret1'] = data[target_column].pct_change().fillna(0)
        data['sma5']  = data[target_column].rolling(window=5).mean().bfill()
        data['sma10'] = data[target_column].rolling(window=10).mean().bfill()
        # Exponential moving averages
        data['ema12'] = data[target_column].ewm(span=12, adjust=False).mean().bfill()
        data['ema26'] = data[target_column].ewm(span=26, adjust=False).mean().bfill()
        # RSI(14)
        delta = data[target_column].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi14'] = 100 - (100 / (1 + rs))
        data['rsi14'] = data['rsi14'].bfill()
        # Bollinger Bands (20,2)
        m20 = data[target_column].rolling(window=20).mean()
        s20 = data[target_column].rolling(window=20).std()
        data['bb_upper'] = (m20 + 2 * s20).bfill()
        data['bb_lower'] = (m20 - 2 * s20).bfill()
        # Bollinger Band width: (upper - lower) / moving average
        width = (data['bb_upper'] - data['bb_lower']) / m20
        # Fill missing values both directions
        width = width.bfill().ffill()
        # Ensure width is a Series (not DataFrame)
        if isinstance(width, pd.DataFrame):
            width = width.iloc[:, 0]
        data['bbwidth'] = width
        # Multi-day returns
        data['ret3'] = data[target_column].pct_change(periods=3).fillna(0)
        data['ret7'] = data[target_column].pct_change(periods=7).fillna(0)
        # Volume ratio
        data['vol_ratio'] = (data['Volume'] / data['Volume'].rolling(window=20).mean()).bfill()
        # Average True Range (ATR: 14-day)
        high_low = data['High'] - data['Low']
        high_prev = (data['High'] - data['Close'].shift(1)).abs()
        low_prev = (data['Low'] - data['Close'].shift(1)).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        data['atr14'] = tr.rolling(window=14).mean().bfill()
        # MACD (12,26,9)
        data['macd'] = data['ema12'] - data['ema26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean().bfill()
        # Sentiment placeholder (to be replaced with actual sentiment scores)
        data['sentiment'] = 0
        # Fill any remaining NaNs in DataFrame (forward/backward and zero-fill)
        data = data.ffill().bfill().fillna(0)

        # Create training data
        # Scale features and target separately, then sanitize NaNs/Infs
        feat_cols = [
            target_column, 'ret1', 'sma5', 'sma10',
            'ema12', 'ema26', 'rsi14',
            'bb_upper', 'bb_lower', 'bbwidth',
            'atr14', 'macd', 'macd_signal',
            'ret3', 'ret7', 'vol_ratio',
            'sentiment'
        ]
        scaled_feats = self.feat_scaler.fit_transform(data[feat_cols].values)
        scaled_feats = np.nan_to_num(scaled_feats, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_price = self.scaler.fit_transform(data[[target_column]].values)
        scaled_price = np.nan_to_num(scaled_price, nan=0.0, posinf=0.0, neginf=0.0)

        x, y = [], []
        for i in range(prediction_days, len(scaled_feats) - future_day + 1):
            x.append(scaled_feats[i - prediction_days:i, :])
            y.append(scaled_price[i + future_day - 1, 0])

        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Default split index to avoid UnboundLocalError
        split_idx = 0

        # Split into train/test and derive split index for sliding windows
        if test_split_date:
            test_data = data[data.index >= test_split_date].copy()
            train_data = data[data.index < test_split_date].copy()
            # Number of training samples for sliding window
            split_idx = max(0, len(train_data) - prediction_days - future_day + 1)
        else:
            split_idx = int(len(x_arr) * (1 - test_size))
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()

        x_train, x_test = x_arr[:split_idx], x_arr[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]
        
        # Inverse transform y for actual prices
        actual_prices = self.inverse_transform(y_test)
        
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "actual_prices": actual_prices,
            "test_dates": test_data.index.tolist(),
            "train_data": train_data,
            "test_data": test_data,
            "scaler": self.scaler,
            "feat_scaler": self.feat_scaler,
            "prediction_days": prediction_days,
            "future_day": future_day
        }
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data.
        
        Args:
            scaled_data: Scaled data
            
        Returns:
            Original scale data
        """
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()