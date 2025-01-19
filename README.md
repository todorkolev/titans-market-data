# Market Titans

Implementation of the Titans paper approach for market data prediction.

## Overview

This project implements the Titans architecture from the paper "Titans: Learning to Memorize at Test Time" for market data prediction. The implementation includes:

- Neural Memory Module with test-time learning
- Multi-head attention mechanism
- Support for multiple assets
- Comprehensive metrics and visualizations

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

## Usage

1. Configure parameters in `configs/config.yaml`
2. Run training:
```bash
python train.py
```

## Features

- Test-time learning with memory updates
- Surprise-based memory mechanism
- Support for multiple market symbols
- Technical indicators as features
- Performance visualization and metrics

## Configuration

Key parameters in `config.yaml`:
- Data parameters (symbols, dates, window sizes)
- Model architecture (dimensions, layers)
- Training parameters (batch size, learning rates)

## Results

The model provides:
- Market predictions with confidence metrics
- Surprise value visualization
- Performance metrics (MSE, MAE, Correlation, Directional Accuracy) 