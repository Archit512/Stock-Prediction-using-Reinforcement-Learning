import gymnasium as gym
import numpy as np
from src.database import db

class CloudPersistentEnv(gym.Env):
    def __init__(self, df):
        super(CloudPersistentEnv, self).__init__()
        self.df = df # Historical data for training
        
        # Observation: [Price_Change, Sentiment, Macro_Panic, Balance_Ratio]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Action: [Type (0:Hold, 1:Buy, 2:Sell), Size (0.0 to 1.0)]
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float32)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        # Calculate Balance Ratio to let the agent know if it's 'winning' or 'losing'
        balance_ratio = self.balance / 10000.0 # Normalized against initial capital
        return np.array([row['price_change'], row['sentiment_score'], row['macro_panic_score'], balance_ratio], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        account = db.get_account_status()
        self.balance = float(account['current_balance'])
        self.shares_held = float(account['total_shares'])
        self.current_step = 0
        self.net_worth = self.balance + (self.shares_held * self.df.iloc[0]['price'])
        return self._get_obs(), {}

    def step(self, action):
        action_type = int(action[0])
        requested_size = action[1]
        current_price = self.df.iloc[self.current_step]['price']

        # --- KELLY CRITERIA (The Risk Governor) ---
        # Map sentiment (-1 to 1) to probability (0 to 1)
        p = (self.df.iloc[self.current_step]['sentiment_score'] + 1) / 2
        kelly_size = max(0, (2 * p - 1)) # Simple Kelly for even odds
        final_size = min(requested_size, kelly_size) # Agent cannot exceed Kelly safety

        # Execute persistent trade
        if action_type == 1: # BUY
            cost = self.balance * final_size
            self.shares_held += cost / current_price
            self.balance -= cost
        elif action_type == 2: # SELL
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        # Sync to Supabase Global Balance
        db.update_account_status(self.balance, self.shares_held)

        # Calculate Reward (Logarithmic Return of Net Worth)
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['price']
        new_net_worth = self.balance + (self.shares_held * new_price)
        
        # Reward is the core of Value Updation
        reward = np.log(new_net_worth / self.net_worth)
        self.net_worth = new_net_worth

        done = self.current_step >= len(self.df) - 2
        return self._get_obs(), reward, done, False, {}