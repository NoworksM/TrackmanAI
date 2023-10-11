import os

import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import PPO

import config
from model import TrackmaniaInputCallback

if os.path.exists("trackmanai_ppo_offline.zip"):
    model = PPO.load("trackmanai_ppo_offline", env=gym.make('TrackmaniaPreTraining-v1'), verbose=1,
                     tensorboard_log=config.tensorboard_log_path)
    reset_timesteps = False
else:
    model = PPO("MultiInputPolicy", 'TrackmaniaPreTraining-v1', verbose=1,
                tensorboard_log=config.tensorboard_log_path)
    reset_timesteps = True

# model.learning_rate = 1.0
# model.policy.optimizer = dadaptation.DAdaptAdam(model.policy.parameters(), lr=model.learning_rate)

while True:
    model.learn(total_timesteps=10000, log_interval=1, reset_num_timesteps=reset_timesteps,
                tb_log_name='trackmania_pretraining',
                callback=[
                    CheckpointCallback(save_freq=5000, save_path=config.checkpoint_path, name_prefix='trackmania_v1',
                                       save_replay_buffer=True, save_vecnormalize=True),
                    TrackmaniaInputCallback()])
    model.save("trackmanai_ppo_offline")
    reset_timeswdswasdteps = False
