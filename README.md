# DASCIENT-RLBot
This Python program uses reinforcement learning (RL) to build a trading bot that can make decisions based on live stock market data. The bot fetches real-time market data using Yahoo Finance, analyzes it, and makes decisions (buy, sell, or hold) using a **Deep Q-Network (DQN)** model.

# RL Trading Bot

## Overview
This Python script uses reinforcement learning (RL) to build a trading bot that can make decisions based on live stock market data. The bot fetches real-time market data using Yahoo Finance, analyzes it, and makes decisions (buy, sell, or hold) using a **Deep Q-Network (DQN)** model.

## Features
- **Reinforcement Learning (RL)**: Trains an agent using DQN to make trading decisions.
- **Live Data**: Fetches live stock market data from Yahoo Finance.
- **Customizable Stock Symbol**: Trade on different stocks by changing the symbol.
- **Debugging Support**: Prints the current status to track agent performance.

## Requirements
- Python 3.x
- TensorFlow (>= 2.x)
- Gym (for creating the trading environment)
- Yahoo Finance (`yfinance`)
- Other dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository: git clone https://github.com/your-username/rl-trading-bot.git cd rl-trading-bot


2. Install required packages: pip install -r requirements
