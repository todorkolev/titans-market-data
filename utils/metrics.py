import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate various performance metrics for predictions.
    
    Args:
        actual: Actual returns
        predicted: Predicted returns
    
    Returns:
        Dictionary of metrics
    """
    # Ensure inputs are 1D arrays
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    
    # Basic metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Correlation
    correlation = np.corrcoef(actual, predicted)[0, 1]
    
    # Directional accuracy
    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)
    directional_accuracy = np.mean(actual_direction == predicted_direction)
    
    # Sharpe ratio (annualized)
    predicted_sharpe = np.sqrt(252) * np.mean(predicted) / np.std(predicted)
    actual_sharpe = np.sqrt(252) * np.mean(actual) / np.std(actual)
    
    # Information ratio
    tracking_error = np.std(actual - predicted) * np.sqrt(252)
    information_ratio = np.mean(actual - predicted) * np.sqrt(252) / tracking_error if tracking_error != 0 else 0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Directional_Accuracy': directional_accuracy,
        'Predicted_Sharpe': predicted_sharpe,
        'Actual_Sharpe': actual_sharpe,
        'Information_Ratio': information_ratio
    } 