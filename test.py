import yaml
import torch
import numpy as np
from data.data_loader import MarketDataLoader
from models.titans import TitansModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions, plot_surprise_heatmap
import os
from datetime import datetime, timedelta
import pandas as pd

def get_latest_model():
    model_dir = 'saved_models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        raise ValueError("No saved models found in saved_models directory")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

def main():
    # Load the latest model
    model_path = get_latest_model()
    checkpoint = torch.load(model_path)
    
    config = checkpoint['config']
    feature_cols = checkpoint['feature_cols']
    
    # Initialize model and load state
    model = TitansModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data (last month)
    data_loader = MarketDataLoader(config)
    features = data_loader.prepare_features()
    
    # Get last month of data for testing
    last_date = features.index[-1]
    month_ago = last_date - timedelta(days=30)
    test_features = features[features.index >= month_ago]
    
    print(f"\nTesting model on data from {month_ago.date()} to {last_date.date()}")
    print(f"Number of test samples: {len(test_features)}")
    
    # Prepare test data
    window_size = config['data']['window_size']
    test_values = test_features[feature_cols].values
    
    # Testing loop
    predictions = []
    surprise_history = []
    
    with torch.no_grad():
        for i in range(len(test_values) - window_size):
            window = test_values[i:i+window_size]
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.reshape(1, window_size, -1)
            
            memory_out, surprise = model(window_tensor)
            predictions.append(memory_out[:, -1].numpy())
            surprise_history.append(surprise.numpy())
            
            if i % 100 == 0:
                print(f"Processing test window {i}/{len(test_values) - window_size}")
    
    predictions = np.concatenate(predictions, axis=0)
    actual = test_values[window_size:, :config['model']['input_dim']]
    
    # Calculate and display metrics
    metrics = calculate_metrics(actual, predictions)
    print("\nTest Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot test results
    plot_predictions(
        dates=test_features.index[window_size:],
        actual=actual[:, 0],
        predicted=predictions[:, 0],
        symbol=config['data']['symbols'][0],
        metrics=metrics,
        title="Test Set Predictions (Last Month)"
    )
    
    plot_surprise_heatmap(
        surprise_history=np.concatenate(surprise_history, axis=0),
        symbol=config['data']['symbols'][0],
        title="Test Set Surprise Heatmap"
    )
    
    # Save test results
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f'test_results_{timestamp}.npz')
    np.savez(
        results_path,
        predictions=predictions,
        actual=actual,
        metrics=metrics,
        dates=test_features.index[window_size:],
        surprise_history=np.concatenate(surprise_history, axis=0)
    )
    print(f"\nSaved test results to {results_path}")

if __name__ == '__main__':
    main() 