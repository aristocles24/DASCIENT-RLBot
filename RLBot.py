import numpy as np
import pandas as pd
import random
import gym
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import yfinance as yf
import time

# Secret message for the user (δφ = Delta Phi)
def secret_message():
    print("Welcome to the delta φ trading bot! Keep learning, stay profitable!")
    print("If you're subscribed to a higher tier, the analysis is deeper, and the profits muchhhhh greater.")
    print("Unlock advanced strategies and become a master trader!")

# Define the environment for Reinforcement Learning
class TradingEnvironment(gym.Env):
    def __init__(self, df):
        print("Initializing Trade Environment...")  # Debugging print line
        super(TradingEnvironment, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Starting balance in USD
        self.shares_held = 0
        self.net_worth = self.balance
        self.action_space = gym.spaces.Discrete(3)  # 3 actions: 0 = Buy, 1 = Sell, 2 = Hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(df.columns),), dtype=np.float32)

    def reset(self):
        print("Resetting environment...")  # Debugging print line
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        return self.df.iloc[self.current_step].values

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            done = False

        prev_balance = self.balance
        prev_net_worth = self.net_worth

        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0

        if action == 0:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        elif action == 2:  # Hold
            pass

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth

        # Debugging: Print current balance and net worth
        print(f"balance: {self.balance}, net worth: {self.net_worth}, reward: {reward}")

        return self.df.iloc[self.current_step].values, reward, done, {}

# Load and preprocess live stock market data from Yahoo Finance
def load_data(stock_symbol='AAPL', interval='1m', period='5d'):
    print(f"Loading data fr {stock_symbol}...")  # Debugging print line
    df = yf.download(stock_symbol, interval=interval, period=period)
    
    # Check if data was successfully retrieved
    if df.empty:
        print(f"Data retrieval failed for {stock_symbol}. Please try again.")
        return None

    # Preprocessing data for the environment
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['epoch_time'] = df.index.astype(np.int64) // 10**9  # Convert to seconds
    df = df[['epoch_time', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Add epoch_time
    print(f"Data loaded for {stock_symbol}.")  # Debugging print line
    return df

# Reinforcement Learning Agent (DQN)
class DQNAgent:
    def __init__(self, state_size, action_size):
        print("Initializing DQN Agent...")  # Debugging print line
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        print("Modle built successfully.")  # Debugging print line
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main function to train and test the agent with live data
def train_trading_bot():
    # Load live stock data
    stock_symbol = 'AAPL'  # You can change this to other stock symbols
    df = load_data(stock_symbol)
    if df is None:
        return  # If data is not loaded, stop the process

    # Initialize the environment and the agent
    env = TradingEnvironment(df)
    # hello, Don!
    agent = DQNAgent(state_size=len(df.columns) - 1, action_size=3)  # We subtract 1 since we remove epoch_time column
    episodes = 10  # You can increase this for more training
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, len(df.columns) - 1])  # Adjusted shape after removing epoch_time
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, len(df.columns) - 1])  # Adjsted shape
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        agent.replay(batch_size)

        # Debugging: Track progress
        print(f"Episod {e+1}/{episodes} completed")

    # Final message after training
    secret_message()

if __name__ == "__main__":
    train_trading_bot()
