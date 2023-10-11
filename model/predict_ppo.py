import os

from stable_baselines3 import PPO
import gymnasium as gym

import config
import model

env = gym.make('Trackmania-v1', render_mode='human')

if os.path.exists("trackmanai_ppo_offline.zip"):
    model = PPO.load("trackmanai_ppo_offline", env=env, verbose=1,
                     tensorboard_log=config.tensorboard_log_path)
    reset_timesteps = False
else:
    raise Exception("No model found")

# model.learning_rate = 1.0
# model.policy.optimizer = dadaptation.DAdaptAdam(model.policy.parameters(), lr=model.learning_rate)

obs, _ = env.reset()

# Run the predict loop until termination
terminated = False
while not terminated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.render()

env.close()
