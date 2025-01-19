import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch
from sklearn.preprocessing import StandardScaler

class MarketDataLoader:
    def __init__(self, config: dict):
        self.symbols = config['data']['symbols']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.window_size = config['data']['window_size']
        self.resolution = config['data']['resolution']
        self.scaler = StandardScaler()
        
    def download_data(self) -> pd.DataFrame:
        """Download market data for all symbols."""
        dfs = []
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, self.start_date, self.end_date, interval=self.resolution)
                if len(df) > 0:  # Only add if we got data
                    df['Symbol'] = symbol
                    dfs.append(df)
            except Exception as e:
                print(f"Failed to download {symbol}: {e}")
        
        if not dfs:
            raise RuntimeError("Failed to download any data")
        
        # Combine and align dates
        combined = pd.concat(dfs)
        combined = combined.reset_index()
        combined = combined.set_index(['Date', 'Symbol']).unstack()
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        combined = combined.stack()
        combined = combined.reset_index()
        combined = combined.set_index('Date')
        
        return combined
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and MACD histogram."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        return upper, ma, lower
    
    def prepare_features(self):
        """
        Prepare features from downloaded data with enhanced technical indicators.
        Returns a DataFrame with features for all symbols.
        """
        features_list = []
        
        for symbol in self.symbols:
            try:
                # Download data
                print(f"Downloading {symbol} data from {self.start_date} to {self.end_date} with {self.resolution} interval...")
                df = yf.download(symbol, self.start_date, self.end_date, interval=self.resolution)
                print(f"Downloaded {len(df)} rows of data")
                if len(df) == 0:
                    print(f"No data available for {symbol}")
                    continue
                    
                # Calculate features
                features = pd.DataFrame(index=df.index)
                
                # Get price series
                close_series = df['Close']
                volume_series = df['Volume']
                high_series = df['High']
                low_series = df['Low']
                
                # Log transform volume
                volume_series = np.log1p(volume_series)
                
                # Returns and volatility
                returns = close_series.pct_change(fill_method=None)
                log_returns = np.log(close_series).diff()
                features['returns'] = returns
                features['log_returns'] = log_returns
                features['volatility'] = returns.rolling(window=20).std()
                
                # Multiple timeframe returns
                for period in [3, 5, 10, 21]:
                    features[f'returns_{period}d'] = close_series.pct_change(period)
                
                # Additional volatility measures
                for window in [10, 30]:
                    features[f'volatility_{window}d'] = returns.rolling(window=window).std()
                
                # Moving averages and crossovers
                ma_periods = [20, 50, 100, 200]
                for period in ma_periods:
                    ma = close_series.rolling(window=period).mean()
                    features[f'ma_{period}'] = (ma / close_series - 1)
                
                # MA crossovers
                for short_ma, long_ma in [(20,50), (50,200)]:
                    short = close_series.rolling(window=short_ma).mean()
                    long = close_series.rolling(window=long_ma).mean()
                    features[f'ma_cross_{short_ma}_{long_ma}'] = (short - long) / long
                
                # RSI with multiple timeframes
                for period in [7, 14, 21]:
                    features[f'rsi_{period}'] = self._calculate_rsi(close_series, period) / 100.0
                
                # MACD
                macd, signal, hist = self._calculate_macd(close_series)
                features['macd'] = macd
                features['macd_signal'] = signal
                features['macd_hist'] = hist
                
                # Bollinger Bands
                for window in [20, 50]:
                    ma = close_series.rolling(window=window).mean()
                    std = close_series.rolling(window=window).std()
                    upper = ma + (std * 2)
                    lower = ma - (std * 2)
                    features[f'bb_upper_{window}'] = (upper / close_series - 1)
                    features[f'bb_lower_{window}'] = (lower / close_series - 1)
                    features[f'bb_position_{window}'] = (close_series - lower) / (upper - lower)
                    features[f'bb_squeeze_{window}'] = (upper - lower).pct_change()
                
                # Volume features
                vol_ma20 = volume_series.rolling(window=20).mean()
                features['volume_ma20'] = vol_ma20
                features['volume_ma20_ratio'] = volume_series.div(vol_ma20).fillna(0)
                features['volume_price_trend'] = volume_series.mul(returns).fillna(0)
                
                # Additional volume features
                for period in [5, 10, 20]:
                    vol_ma = volume_series.rolling(window=period).mean()
                    features[f'volume_ma_{period}'] = vol_ma
                    features[f'volume_ma_{period}_ratio'] = volume_series.div(vol_ma).fillna(0)
                
                # Price momentum
                for period in [3, 5, 10, 21]:
                    features[f'momentum_{period}d'] = (close_series.diff(period) / close_series.shift(period))
                
                # Rate of Change (ROC)
                for period in [5, 10, 20]:
                    features[f'roc_{period}d'] = ((close_series - close_series.shift(period)) / close_series.shift(period))
                
                # High-Low range features
                features['daily_range'] = (high_series - low_series) / close_series
                features['range_ma20'] = features['daily_range'].rolling(window=20).mean()
                
                # True Range and ATR
                tr = pd.concat([
                    high_series - low_series,
                    abs(high_series - close_series.shift(1)),
                    abs(low_series - close_series.shift(1))
                ], axis=1).max(axis=1)
                
                for period in [14, 20]:
                    features[f'atr_{period}'] = tr.rolling(window=period).mean()
                    features[f'atr_ratio_{period}'] = tr / tr.rolling(window=period).mean()
                
                # Price position relative to moving averages
                for ma in ma_periods:
                    ma_series = close_series.rolling(window=ma).mean()
                    features[f'price_to_ma_{ma}'] = (close_series / ma_series - 1)
                
                # Add symbol identifier
                features['Symbol'] = symbol
                
                # Drop NaN values
                features = features.dropna()
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        if not features_list:
            raise RuntimeError("No features could be calculated for any symbol")
        
        # Combine features from all symbols
        all_features = pd.concat(features_list)
        all_features = all_features.sort_index()
        
        # Normalize numerical features
        numeric_columns = all_features.select_dtypes(include=[np.number]).columns
        symbol_column = all_features['Symbol']
        all_features[numeric_columns] = self.scaler.fit_transform(all_features[numeric_columns])
        all_features['Symbol'] = symbol_column
        
        return all_features 