import os

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

import config
from model import TrackmaniaInputCallback
from model.reward.trackmania_2020_reward_calculator_race_v1 import TrackmaniaRewardCalculatorRace

model_name = 'trackmanai_recurrent_ppo'

env = gym.make('Trackmania-v1', reward_calculator=TrackmaniaRewardCalculatorRace())

if os.path.exists(model_name + '.zip'):
    model = RecurrentPPO.load(model_name,
                              env=env,
                              verbose=1,
                              tensorboard_log=config.tensorboard_log_path)
    reset_timesteps = False
else:
    model = RecurrentPPO('MultiInputLstmPolicy', env=env, verbose=1,
                         tensorboard_log=config.tensorboard_log_path, n_steps=4096)
    reset_timesteps = True

# model.learning_rate = 1.0
# model.policy.optimizer = dadaptation.DAdaptAdam(model.policy.parameters(), lr=model.learning_rate)

while True:
    model.learn(total_timesteps=20000, log_interval=1, reset_num_timesteps=reset_timesteps,
                tb_log_name=model_name,
                callback=[
                    CheckpointCallback(save_freq=10000, save_path=config.checkpoint_path,
                                       name_prefix='trackmania_recurrent_ppo_v1',
                                       save_replay_buffer=True, save_vecnormalize=True),
                    TrackmaniaInputCallback(), ProgressBarCallback()])
    model.save(model_name)
    reset_timesteps = False
