from stable_baselines3 import PPO
from src.rl_env.cloud_env import CloudPersistentEnv
from src.database import db
from loguru import logger

def train_brain():
    # 1. Fetch historical logs from Supabase
    logger.info("📥 Downloading market history for training...")
    df = db.get_training_data()
    
    if len(df) < 100:
        logger.error("❌ Not enough data in Supabase yet. Run Coordinator for a few days!")
        return

    # 2. Setup Environment
    env = CloudPersistentEnv(df)

    # 3. Initialize PPO Agent (The Actor-Critic)
    # ent_coef: Exploration vs Exploitation balance
    # clip_range: Safety limit for policy updates
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        ent_coef=0.01, 
        clip_range=0.2,
        gamma=0.99
    )

    # 4. Value Updation: Agent plays the 'game' of your past trades
    logger.info("🧠 Training PPO Agent on MNNIT Market Data...")
    model.learn(total_timesteps=20000)

    # 5. Save the 'Brain' for Google Cloud Deployment
    model.save("ppo_trading_model_v1")
    logger.success("✅ Training Complete! Model saved as ppo_trading_model_v1.zip")

if __name__ == "__main__":
    train_brain()