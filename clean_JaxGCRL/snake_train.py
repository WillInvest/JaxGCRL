import os
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from envs.snake_cnn import SnakeEnv
from utils.combine_video import combine_videos

from datetime import datetime

# Create a function to instantiate the environment
def make_env():
    def _init():
        env = SnakeEnv(render_mode=False)  # No rendering during training
        return env
    return _init

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """This method is called at every step."""
        # Log the info metrics from the environment to WandB
        info = self.locals['infos'][0]  # Access the 'info' from the first environment
        done = self.locals['dones'][0]  # Access the 'done' from the first environment
        # log score and steps when done is True
        if done:
            wandb.log({
                "score": info.get("score"),
                "steps": info.get("steps")
            })
 
        return True


def main():
    
    # Initialize WandB
    wandb.init(
        project="ppo_snake",
        config={
            "env_name": "SnakeEnv",
            "algo": "PPO",
            "total_timesteps": 100000,
            "num_envs": 32,
            "policy_type": "MlpPolicy",
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        reinit=False
    )
    # Create 32 environments for parallel training
    num_envs = 32
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        envs,
        gamma=0.95,
        verbose=0,
        n_steps=2048,  # Number of steps per update
        batch_size=64,
        n_epochs=10
    )

    # Create a Checkpoint callback to save the model every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save the model every 5000 steps
        save_path='./checkpoints/',  # Directory to save the model
        name_prefix='ppo_snake'
    )

    # Create a WandB callback
    wandb_callback = WandbCallback()

    # Train the model using multiple environments with all callbacks
    model.learn(
        total_timesteps=int(1e10),
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=True
    )

    # Save the trained model
    model.save("ppo_snake_parallel")

    # Finish the WandB run
    wandb.finish()
    
    envs.close()


def display_snake(model_path, root_dir, pattern="snake_game"):
    # Load the model and test it with a single environment
    model = PPO.load(model_path)

    # Test the model with a single environment
    env = SnakeEnv(render_mode=True)  # Enable rendering for testing
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
    
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(root_dir, f"snake_all_{time_stamp}.avi")
    combine_videos(output_file=output_dir, 
                   root_dir=root_dir,
                   pattern=pattern)


if __name__ == "__main__":
    main()
    root_dir = '/home/shiftpub/JaxGCRL/clean_JaxGCRL/video'
    pattern = 'snake_game'
    model_path = "/home/shiftpub/JaxGCRL/clean_JaxGCRL/checkpoints/ppo_snake_480000_steps.zip"
    display_snake(model_path, root_dir, pattern)