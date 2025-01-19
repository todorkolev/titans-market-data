import yaml
import torch
from data.data_loader import MarketDataLoader
from models.titans import TitansModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions, plot_surprise_heatmap
import numpy as np
import os
from datetime import datetime

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize data loader and model
    data_loader = MarketDataLoader(config)
    features = data_loader.prepare_features()
    
    # Split data into train and validation (keep last month for testing)
    train_cutoff = len(features) - int(len(features) * 0.2)  # 20% for validation
    train_features = features.iloc[:train_cutoff]
    val_features = features.iloc[train_cutoff:]
    
    model = TitansModel(config)
    
    # Training loop
    window_size = config['data']['window_size']
    feature_window = config['data']['feature_window']
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # Get numeric feature columns
    feature_cols = ['returns', 'volatility', 'ma50', 'ma200', 'rsi']
    train_values = train_features[feature_cols].values
    val_values = val_features[feature_cols].values
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_predictions = []
        train_surprise_history = []
        
        for i in range(len(train_values) - window_size):
            window = train_values[i:i+window_size]
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.reshape(1, window_size, -1)
            
            memory_out, surprise = model(window_tensor)
            train_predictions.append(memory_out[:, -1].detach().numpy())
            train_surprise_history.append(surprise.detach().numpy())
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Processing window {i}/{len(train_values) - window_size}")
        
        # Validation step
        model.eval()
        val_predictions = []
        val_surprise_history = []
        
        with torch.no_grad():
            for i in range(len(val_values) - window_size):
                window = val_values[i:i+window_size]
                window_tensor = torch.tensor(window, dtype=torch.float32)
                window_tensor = window_tensor.reshape(1, window_size, -1)
                
                memory_out, surprise = model(window_tensor)
                val_predictions.append(memory_out[:, -1].numpy())
                val_surprise_history.append(surprise.numpy())
        
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_actual = val_values[window_size:, :config['model']['input_dim']]
        val_metrics = calculate_metrics(val_actual, val_predictions)
        val_loss = val_metrics['mse']
        
        print(f"\nEpoch {epoch} Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                os.remove(best_model_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f"saved_models/titans_model_{timestamp}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'feature_cols': feature_cols
            }, best_model_path)
            print(f"\nSaved new best model to {best_model_path}")
    
    # Final evaluation and plotting
    model.eval()
    final_predictions = []
    final_surprise_history = []
    
    with torch.no_grad():
        for i in range(len(val_values) - window_size):
            window = val_values[i:i+window_size]
            window_tensor = torch.tensor(window, dtype=torch.float32)
            window_tensor = window_tensor.reshape(1, window_size, -1)
            
            memory_out, surprise = model(window_tensor)
            final_predictions.append(memory_out[:, -1].numpy())
            final_surprise_history.append(surprise.numpy())
    
    final_predictions = np.concatenate(final_predictions, axis=0)
    final_actual = val_values[window_size:, :config['model']['input_dim']]
    
    # Plot final results
    plot_predictions(
        dates=val_features.index[window_size:],
        actual=final_actual[:, 0],
        predicted=final_predictions[:, 0],
        symbol=config['data']['symbols'][0],
        metrics=calculate_metrics(final_actual, final_predictions)
    )
    
    plot_surprise_heatmap(
        surprise_history=np.concatenate(final_surprise_history, axis=0),
        symbol=config['data']['symbols'][0]
    )

if __name__ == '__main__':
    main() 