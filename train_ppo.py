import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from stable_baselines3 import PPO
from src.rl_env.cloud_env import CloudPersistentEnv
from src.database import db
from loguru import logger
import os

def train_brain():
    try:
        # 1. Fetch historical logs
        logger.info("Downloading market history for training...")
        df = db.get_training_data()
        
        # Verify data exists
        if df is None or df.empty:
            logger.error("DataFrame is empty! Check db.get_training_data() query.")
            return
            
        if len(df) < 100:
            logger.error(f"Only {len(df)} rows found. Need at least 100.")
            return

        # 2. Setup Environment (FLAGGED AS TRAINING)
        logger.info("Initializing Sandbox Environment...")
        env = CloudPersistentEnv(df, is_training=True)

        # 3. Initialize PPO Agent
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            learning_rate=3e-4,
            ent_coef=0.05, # Changed from 0.01 to encourage more exploration during training
            clip_range=0.2,
            gamma=0.99
        )

        # 4. Train
        logger.info("Starting local PPO training...")
        model.learn(total_timesteps=150000) # Chnaged from 100k to 150k for better convergence

        # 5. Save the Model
        os.makedirs("models", exist_ok=True)
        save_path = "models/ppo_trading_model_v2"
        
        model.save(save_path)
        logger.success(f"Training Complete! Model saved to {save_path}.zip")

    except Exception as e:
        logger.error(f"Fatal error during training: {e}")

if __name__ == "__main__":
    train_brain()