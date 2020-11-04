import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import stable_baselines
import tensorflow as tf

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647

MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 100


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        # super(StockTradingEnv, self).__init__()

        self.df = df
        self.MAX_SHARE_PRICE = max(df[['Open','High', 'Low', 'Close']].max())
        self.MIN_SHARE_PRICE = min(df[['Open','High', 'Low', 'Close']].min())
        self.MAX_VOLUME = max(df[['Volume']].max())
        self.MIN_VOLUME = min(df[['Volume']].min())

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        #self.action_space = spaces.Box(
        #    low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        self.action_space = spaces.Discrete(3)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
 
        self.T = 5 # Length of each episode

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.append([
            (self.df.loc[self.current_step: self.current_step +
                        0, 'Open'].values - self.MIN_SHARE_PRICE) / (self.MAX_SHARE_PRICE - self.MIN_SHARE_PRICE)],
            #[(self.df.loc[self.current_step: self.current_step +
                       # 1, 'High'].values - self.MIN_SHARE_PRICE) / (self.MAX_SHARE_PRICE - self.MIN_SHARE_PRICE),
            #(self.df.loc[self.current_step: self.current_step +
                       # 1, 'Low'].values - self.MIN_SHARE_PRICE) / (self.MAX_SHARE_PRICE - self.MIN_SHARE_PRICE),
            [(self.df.loc[self.current_step: self.current_step +
                        0, 'Close'].values - self.MIN_SHARE_PRICE) / (self.MAX_SHARE_PRICE - self.MIN_SHARE_PRICE),
            (self.df.loc[self.current_step: self.current_step +
                        0, 'Volume'].values - self.MIN_VOLUME) / (self.MAX_VOLUME - self.MIN_VOLUME),
            ]) # Volume is the amount of shares that changed hands during a given day
        
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            # self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / self.MAX_SHARE_PRICE,
            # self.total_shares_sold / MAX_NUM_SHARES,
            # self.total_sales_value / (MAX_NUM_SHARES * self.MAX_SHARE_PRICE),
        ])
        
        return self.discretize(obs)

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        print(current_price)

        if action == 0:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            if total_possible >= 1:
                shares_bought = 1
            else:
                shares_bought = 0
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            if self.shares_held+shares_bought != 0:
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)
            else:
                self.cost_basis = 0
            self.shares_held += 1

        elif action == 1:
            # Sell amount % of shares held
            shares_sold = 1 #int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            # self.total_shares_sold += shares_sold
            # self.total_sales_value += shares_sold * current_price

        else:# do nothing
            pass 
            
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.timestep += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = round(self.net_worth, 2) # * delay_modifier
        done = self.timestep >= self.T # Allow a trading period of 5 days

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)
        self.timestep = 0

        state = self._next_observation()
        
        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

    def discretize(self,array):
        for i in range(array.shape[0]):
            if array[i]>=0 and array[i] <0.3:
                array[i] = int(0)
            elif array[i]>=0.3 and array[i] <0.7:
                array[i] = int(1)
            elif array[i]>=0.7:
                array[i] = int(2)

        return array.astype(dtype=np.int32, copy=False)