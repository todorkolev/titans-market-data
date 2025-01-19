import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.data_loader import MarketDataLoader
from models.titans import TitansModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions, plot_surprise_heatmap
import numpy as np
import os
from datetime import datetime

def train_epoch(model, data, window_size, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    predictions = []
    surprise_history = []
    
    for i in range(len(data) - window_size - 1):  # -1 to have target
        # Get input window and target
        window = data[i:i+window_size]
        target = data[i+1:i+window_size+1]  # Next step prediction
        
        # Convert to tensors
        window_tensor = torch.tensor(window, dtype=torch.float32, device=device)
        target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
        
        # Add batch dimension
        window_tensor = window_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)
        
        # Forward pass
        optimizer.zero_grad()
        memory_out, surprise = model(window_tensor)
        
        # Calculate loss
        loss = criterion(memory_out, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        predictions.append(memory_out[:, -1].detach().cpu().numpy())
        surprise_history.append(surprise.detach().cpu().numpy())
        
        if i % 100 == 0:
            print(f"Processing window {i}/{len(data) - window_size - 1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / (len(data) - window_size - 1)
    predictions = np.concatenate(predictions, axis=0)
    surprise_history = np.concatenate(surprise_history, axis=0)
    
    return avg_loss, predictions, surprise_history

def validate(model, data, window_size, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    surprise_history = []
    
    with torch.no_grad():
        for i in range(len(data) - window_size - 1):
            window = data[i:i+window_size]
            target = data[i+1:i+window_size+1]
            
            window_tensor = torch.tensor(window, dtype=torch.float32, device=device)
            target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
            
            window_tensor = window_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
            
            memory_out, surprise = model(window_tensor)
            loss = criterion(memory_out, target_tensor)
            
            total_loss += loss.item()
            predictions.append(memory_out[:, -1].cpu().numpy())
            surprise_history.append(surprise.cpu().numpy())
    
    avg_loss = total_loss / (len(data) - window_size - 1)
    predictions = np.concatenate(predictions, axis=0)
    surprise_history = np.concatenate(surprise_history, axis=0)
    
    return avg_loss, predictions, surprise_history

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Initialize data loader and model
    data_loader = MarketDataLoader(config)
    features = data_loader.prepare_features()
    
    # Split data into train and validation
    train_cutoff = len(features) - int(len(features) * 0.2)  # 20% for validation
    train_features = features.iloc[:train_cutoff]
    val_features = features.iloc[train_cutoff:]
    
    # Get numeric feature columns
    feature_cols = ['returns', 'volatility', 'ma50', 'ma200', 'rsi']
    train_values = train_features[feature_cols].values
    val_values = val_features[feature_cols].values
    
    # Initialize model and move to device
    model = TitansModel(config).to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training parameters
    window_size = config['data']['window_size']
    best_val_loss = float('inf')
    best_model_path = None
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Training phase
        train_loss, train_predictions, train_surprise = train_epoch(
            model, train_values, window_size, optimizer, criterion, device
        )
        
        # Validation phase
        val_loss, val_predictions, val_surprise = validate(
            model, val_values, window_size, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate metrics
        train_metrics = calculate_metrics(
            train_values[window_size+1:, :config['model']['input_dim']],
            train_predictions
        )
        
        val_metrics = calculate_metrics(
            val_values[window_size+1:, :config['model']['input_dim']],
            val_predictions
        )
        
        # Print metrics
        print(f"\nTraining Loss: {train_loss:.4f}")
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nValidation Loss: {val_loss:.4f}")
        print("Validation Metrics:")
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
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'val_metrics': val_metrics,
                'feature_cols': feature_cols
            }, best_model_path)
            print(f"\nSaved new best model to {best_model_path}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_loss, final_predictions, final_surprise = validate(
        model, val_values, window_size, criterion, device
    )
    
    final_metrics = calculate_metrics(
        val_values[window_size+1:, :config['model']['input_dim']],
        final_predictions
    )
    
    # Plot final results
    plot_predictions(
        dates=val_features.index[window_size+1:],
        actual=val_values[window_size+1:, 0],  # First feature (returns)
        predicted=final_predictions[:, 0],
        symbol=config['data']['symbols'][0],
        metrics=final_metrics
    )
    
    plot_surprise_heatmap(
        surprise_history=final_surprise,
        symbol=config['data']['symbols'][0]
    )

if __name__ == '__main__':
    main() 