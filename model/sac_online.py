import os

import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
from stable_baselines3 import SAC

import config
from model.reward import TrackmaniaRewardCalculatorRace


def learn_sac_online(env_name='Trackmania-v1', steps=10000, rollout_steps=4096, model_name='trackmanai_sac_online',
                     frame_stacking=1):
    env = gym.make(env_name, reward_calculator=TrackmaniaRewardCalculatorRace())

    policy = None

    # Flatten and FrameStack the environment for referential past data for the model
    if frame_stacking > 1:
        env = FlattenObservation(env)
        env = FrameStack(env, frame_stacking)
        policy = 'MlpPolicy'
    else:
        policy = 'MultiInputPolicy'

    if os.path.exists(model_name + ".zip"):
        model = SAC.load(model_name, env=env, verbose=1,
                         tensorboard_log=config.tensorboard_log_path)
        reset_timesteps = False
    else:
        model = SAC(policy, env=env, verbose=1,
                    tensorboard_log=config.tensorboard_log_path, rollout_steps=rollout_steps)
        reset_timesteps = True

    model.learn(total_timesteps=steps, log_interval=1, reset_num_timesteps=reset_timesteps, )
