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
        self.scaler = StandardScaler()
        
    def download_data(self) -> pd.DataFrame:
        """Download market data for all symbols."""
        dfs = []
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, self.start_date, self.end_date)
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
                df = yf.download(symbol, self.start_date, self.end_date)
                if len(df) == 0:
                    print(f"No data available for {symbol}")
                    continue
                    
                # Calculate features
                features = pd.DataFrame(index=df.index)
                
                # Required features for training
                close_series = df['Close']
                volume_series = df['Volume']
                
                # Log transform volume
                volume_series = np.log1p(volume_series)
                
                # Returns and volatility
                returns = close_series.pct_change(fill_method=None)
                features['returns'] = returns
                features['volatility'] = returns.rolling(window=20).std()
                
                # Moving averages
                ma50 = close_series.rolling(window=50).mean()
                ma200 = close_series.rolling(window=200).mean()
                features['ma50'] = (ma50 / close_series - 1)
                features['ma200'] = (ma200 / close_series - 1)
                
                # RSI
                features['rsi'] = self._calculate_rsi(close_series) / 100.0  # Normalize RSI to [0,1]
                
                # Volume features
                vol_ma20 = volume_series.rolling(window=20).mean()
                features['volume_ma20'] = vol_ma20
                features['volume_ma20_ratio'] = volume_series.div(vol_ma20).fillna(0)
                features['volume_price_trend'] = volume_series.mul(returns).fillna(0)
                
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