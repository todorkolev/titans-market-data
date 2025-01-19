# Market Titans

## Overview

This project implements the Titans architecture from ["Titans: Learning to Memorize at Test Time"](https://arxiv.org/abs/2501.00663v1) (Behrouz et al., 2024) for market prediction. The Titans model introduces a novel neural long-term memory module that can effectively handle context windows of over 2M tokens while maintaining fast parallel training and inference, making it particularly interesting for analyzing extensive market history and patterns.

The implementation includes:

- Neural Memory Module with test-time learning
- Multi-head attention mechanism
- Support for multiple assets
- Comprehensive metrics and visualizations
- Automated model saving and testing pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `data/`: Data loading and feature engineering
- `models/`: Neural network models implementation
  - `memory_module.py`: Neural Memory Module
  - `attention.py`: Multi-head attention
  - `titans.py`: Main Titans model
- `utils/`: Utility functions for metrics and visualization
- `configs/`: Configuration files
- `train.py`: Main training script
- `test.py`: Model evaluation script
- `saved_models/`: Directory for saved model checkpoints (gitignored)
- `test_results/`: Directory for test results and visualizations (gitignored)

## Usage

1. Configure parameters in `configs/config.yaml`
2. Train the model:
```bash
python train.py
```
3. Test the latest model:
```bash
python test.py
```

## Training Process

The training script (`train.py`):
- Splits data into training and validation sets
- Trains the model for specified epochs
- Automatically saves the best model based on validation loss
- Generates validation metrics and visualizations
- Saves model checkpoints in `saved_models/` directory

## Testing Process

The testing script (`test.py`):
- Automatically loads the latest trained model
- Evaluates on the last month of market data
- Generates comprehensive test metrics
- Creates prediction visualizations
- Saves test results in `test_results/` directory

## Features

- Test-time learning with memory updates
- Surprise-based memory mechanism
- Support for multiple market symbols
- Technical indicators as features
- Performance visualization and metrics
- Automated model checkpointing
- Separate validation and test sets

## Configuration

Key parameters in `config.yaml`:
- Data parameters (symbols, dates, window sizes)
- Model architecture (dimensions, layers)
- Training parameters (batch size, learning rates, epochs)

## Results

The model provides:
- Market predictions with confidence metrics
- Surprise value visualization
- Performance metrics (MSE, MAE, Correlation, Directional Accuracy)
- Saved model checkpoints for best performing models
- Comprehensive test results and visualizations 

## Cloud Deployment Options

### Google Colab (Easiest)
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Clone the repository:
```bash
!git clone https://github.com/todorkolev/titans-market-data.git
```
4. Install the dependencies:
```bash
!cd titans-market-data && pip install -r requirements.txt
```
5. Run the training:
```bash
!cd titans-market-data && python train.py
```

### AWS EC2
1. Launch an EC2 instance (recommended: g4dn.xlarge for GPU support)
2. Connect to your instance:
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@your-instance-ip
```
3. Install dependencies:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip git
```
4. Clone and run:
```bash
git clone https://github.com/todorkolev/titans-market-data.git
cd titans-market-data
pip install -r requirements.txt
python train.py
```

For persistent storage, consider attaching an EBS volume to your EC2 instance. 