import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple
import torch

class MarketDataLoader:
    def __init__(self, config: dict):
        self.symbols = config['data']['symbols']
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.window_size = config['data']['window_size']
        
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
    
    def prepare_features(self):
        """
        Prepare features from downloaded data.
        Returns a DataFrame with features for all symbols.
        """
        # Download data for each symbol
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
                
                # Returns
                features['returns'] = df['Close'].pct_change(fill_method=None)
                
                # Volatility
                features['volatility'] = features['returns'].rolling(window=20).std()
                
                # Moving averages
                features['ma50'] = (df['Close'].rolling(window=50).mean() / df['Close'] - 1)
                features['ma200'] = (df['Close'].rolling(window=200).mean() / df['Close'] - 1)
                
                # RSI
                features['rsi'] = self._calculate_rsi(df['Close'])
                
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
        
        # Normalize numeric features
        numeric_cols = ['returns', 'volatility', 'ma50', 'ma200', 'rsi']
        all_features[numeric_cols] = (all_features[numeric_cols] - all_features[numeric_cols].mean()) / (all_features[numeric_cols].std() + 1e-8)
        
        return all_features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs)) 