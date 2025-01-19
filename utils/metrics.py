import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'correlation': np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1],
        'directional_accuracy': np.mean(np.sign(y_true) == np.sign(y_pred))
    } 