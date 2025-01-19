import yaml
import torch
from data.data_loader import MarketDataLoader
from models.titans import TitansModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions, plot_surprise_heatmap
import numpy as np

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader and model
    data_loader = MarketDataLoader(config)
    features = data_loader.prepare_features()
    
    model = TitansModel(config)
    
    # Training loop
    window_size = config['data']['window_size']
    feature_window = config['data']['feature_window']
    
    predictions = []
    surprise_history = []
    
    # Get numeric feature columns
    feature_cols = ['returns', 'volatility', 'ma50', 'ma200', 'rsi']
    feature_values = features[feature_cols].values
    
    for i in range(len(feature_values) - window_size):
        # Get current window of features
        window = feature_values[i:i+window_size]
        window_tensor = torch.tensor(window, dtype=torch.float32)
        
        # Reshape input to match model's expected dimensions
        window_tensor = window_tensor.reshape(1, window_size, -1)  # [batch_size, seq_len, input_dim]
        
        # Forward pass
        memory_out, surprise = model(window_tensor)
        predictions.append(memory_out[:, -1].detach().numpy())  # Store last prediction
        surprise_history.append(surprise.detach().numpy())
        
        # Print progress
        if i % 100 == 0:
            print(f"Processing window {i}/{len(feature_values) - window_size}")
            print(f"Average surprise: {surprise.mean().item():.4f}")
    
    # Convert predictions and get actual values
    predictions = np.concatenate(predictions, axis=0)
    actual = feature_values[window_size:, :config['model']['input_dim']]
    
    # Calculate metrics and plot results
    metrics = calculate_metrics(actual, predictions)
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions vs actual
    plot_predictions(
        dates=features.index[window_size:],
        actual=actual[:, 0],  # Use first feature (returns)
        predicted=predictions[:, 0],
        symbol=config['data']['symbols'][0],
        metrics=metrics
    )
    
    # Plot surprise heatmap
    plot_surprise_heatmap(
        surprise_history=np.concatenate(surprise_history, axis=0),
        symbol=config['data']['symbols'][0]
    )

if __name__ == '__main__':
    main() 