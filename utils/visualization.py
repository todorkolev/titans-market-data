import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import pandas as pd

def plot_predictions(dates: List, actual: List, predicted: List, 
                    symbol: str, metrics: Dict[str, float]):
    """Plot actual vs predicted values with metrics."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', alpha=0.7)
    plt.plot(dates, predicted, label='Predicted', alpha=0.7)
    
    plt.title(f'Market Predictions for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    
    # Add metrics text box
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_surprise_heatmap(surprise_history: np.ndarray, symbol: str):
    """Plot heatmap of surprise values over time."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(surprise_history.T, cmap='RdBu', center=0)
    plt.title(f'Surprise Values Over Time - {symbol}')
    plt.xlabel('Time Step')
    plt.ylabel('Memory Dimension')
    plt.tight_layout()
    plt.show() 