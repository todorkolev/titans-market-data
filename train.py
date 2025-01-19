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
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_feature_importance(model, val_data, window_size, device):
    model.eval()
    feature_importances = torch.zeros(model.input_dim).to(device)
    valid_calculations = 0

    # Get a sample window from validation data
    window = val_data[:window_size]
    target = val_data[1:window_size+1]  # Next step prediction
    
    # Convert to tensors
    window_tensor = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    window_tensor.requires_grad_(True)
    
    # Forward pass
    model.zero_grad()
    output, _ = model(window_tensor)
    loss = F.mse_loss(output, torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0))
    
    # Backward pass
    loss.backward()
    
    # Check if gradients were computed
    if window_tensor.grad is not None:
        # Take absolute mean across batch dimension
        importance = torch.abs(window_tensor.grad).mean(dim=0)
        # Take mean across time dimension
        importance = importance.mean(dim=0)
        feature_importances += importance
        valid_calculations += 1

    if valid_calculations > 0:
        return feature_importances
    else:
        print("Warning: No valid feature importance calculations were made")
        return torch.ones(model.input_dim) / model.input_dim

def plot_feature_importance(importances, feature_cols, top_k=20):
    """Plot feature importance."""
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(importances)
    pos = np.arange(min(top_k, len(sorted_idx))) + .5
    
    # Plot top-k features
    top_features = sorted_idx[-top_k:][::-1]
    plt.barh(pos, importances[top_features])
    plt.yticks(pos, [feature_cols[i] for i in top_features])
    plt.xlabel('Relative Importance')
    plt.title(f'Top {top_k} Most Important Features')
    plt.tight_layout()
    plt.savefig('test_results/feature_importance.png')
    plt.close()

def train_epoch(model, data, window_size, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    predictions = []
    surprise_history = []
    grad_norms = []
    
    for i in range(len(data) - window_size - 1):  # -1 to have target
        try:
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
            
            # Calculate gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norms.append(grad_norm.item())
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            predictions.append(memory_out[:, -1].detach().cpu().numpy())
            surprise_history.append(surprise.detach().cpu().numpy())
            
            if i % 100 == 0:
                print(f"Processing window {i}/{len(data) - window_size - 1}, Loss: {loss.item():.4f}, Grad norm: {grad_norm.item():.4f}")
        
        except RuntimeError as e:
            print(f"Error in batch {i}: {str(e)}")
            continue
    
    if len(predictions) == 0:
        raise RuntimeError("No valid predictions were made during training")
    
    avg_loss = total_loss / len(predictions)
    predictions = np.concatenate(predictions, axis=0)
    surprise_history = np.concatenate(surprise_history, axis=0)
    
    return avg_loss, predictions, surprise_history, np.mean(grad_norms)

def validate(model, data, window_size, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    surprise_history = []
    
    with torch.no_grad():
        for i in range(len(data) - window_size - 1):
            try:
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
            
            except RuntimeError as e:
                print(f"Error in validation batch {i}: {str(e)}")
                continue
    
    if len(predictions) == 0:
        raise RuntimeError("No valid predictions were made during validation")
    
    avg_loss = total_loss / len(predictions)
    predictions = np.concatenate(predictions, axis=0)
    surprise_history = np.concatenate(surprise_history, axis=0)
    
    return avg_loss, predictions, surprise_history

def load_checkpoint(path):
    """Load checkpoint with proper settings."""
    return torch.load(path, map_location='cpu', weights_only=True)

def plot_surprise_heatmap(surprise_history, symbol):
    """Plot heatmap of prediction surprises over time."""
    if len(surprise_history.shape) == 3:
        # Take mean across batch dimension if needed
        surprise_history = surprise_history.mean(axis=0)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    sns.heatmap(surprise_history, cmap='RdBu', center=0)
    plt.title(f'Prediction Surprise Heatmap for {symbol}')
    plt.xlabel('Time Steps')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(f'test_results/surprise_heatmap_{symbol}.png')
    plt.close()

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('test_results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data loader and model
    data_loader = MarketDataLoader(config)
    features = data_loader.prepare_features()
    
    # Get numeric feature columns
    feature_cols = [col for col in features.columns if col != 'Symbol']
    print(f"\nUsing {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i}. {col}")
    print("\n")
    
    # Split data into train and validation by date
    dates = features.index.unique()
    train_dates = dates[:-int(len(dates) * 0.2)]
    val_dates = dates[-int(len(dates) * 0.2):]
    
    train_features = features[features.index.isin(train_dates)]
    val_features = features[features.index.isin(val_dates)]
    
    print(f"Training data from {train_dates[0]} to {train_dates[-1]}")
    print(f"Validation data from {val_dates[0]} to {val_dates[-1]}\n")
    
    train_values = train_features[feature_cols].values
    val_values = val_features[feature_cols].values
    
    # Update config with actual input dimension
    config['model']['input_dim'] = len(feature_cols)
    print(f"Updated model input dimension to {len(feature_cols)}\n")
    
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
        patience=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training parameters
    window_size = config['data']['window_size']
    best_val_loss = float('inf')
    best_model_path = None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_metrics': [], 'val_metrics': [],
        'grad_norms': []
    }
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Training phase
        train_loss, train_predictions, train_surprise, grad_norm = train_epoch(
            model, train_values, window_size, optimizer, criterion, device
        )
        history['grad_norms'].append(grad_norm)
        
        # Validation phase
        val_loss, val_predictions, val_surprise = validate(
            model, val_values, window_size, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        
        # Calculate metrics
        train_metrics = calculate_metrics(
            train_values[window_size+1:, 0],
            train_predictions[:, 0]
        )
        
        val_metrics = calculate_metrics(
            val_values[window_size+1:, 0],
            val_predictions[:, 0]
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
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
            torch.save(
                model.state_dict(),
                best_model_path
            )
            print(f"\nSaved new best model to {best_model_path}")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    # Calculate and plot feature importance
    print("\nCalculating feature importance...")
    importances = calculate_feature_importance(model, val_values, window_size, device)
    plot_feature_importance(importances.cpu().numpy(), feature_cols)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('test_results/training_history.png')
    plt.close()
    
    # Plot gradient norms
    plt.figure(figsize=(12, 6))
    plt.plot(history['grad_norms'], label='Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm History')
    plt.legend()
    plt.savefig('test_results/gradient_norms.png')
    plt.close()
    
    # Load best model for final evaluation
    state_dict = load_checkpoint(best_model_path)
    model.load_state_dict(state_dict)
    
    # Final evaluation
    final_loss, final_predictions, final_surprise = validate(
        model, val_values, window_size, criterion, device
    )
    
    final_metrics = calculate_metrics(
        val_values[window_size+1:, 0],
        final_predictions[:, 0]
    )
    
    # Plot final results
    plot_predictions(
        dates=val_features.index[window_size+1:],
        actual=val_values[window_size+1:, 0],
        predicted=final_predictions[:, 0],
        symbol=config['data']['symbols'][0],
        metrics=final_metrics
    )
    
    plot_surprise_heatmap(
        surprise_history=final_surprise,
        symbol=config['data']['symbols'][0]
    )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise 