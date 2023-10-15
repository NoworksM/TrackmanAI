import os

import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback

import config
from model import TrackmaniaInputCallback
from model.reward.trackmania_2020_reward_calculator_race_v1 import TrackmaniaRewardCalculatorRace


def learn_recurrent_ppo_online(env_name='Trackmania-v1', steps=10000, rollout_steps=4096,
                               model_name='trackmanai_recurrent_ppo',
                               frame_stacking=1):
    env = gym.make(env_name, reward_calculator=TrackmaniaRewardCalculatorRace())

    # Flatten and FrameStack the environment for referential past data for the model
    if frame_stacking > 1:
        env = FlattenObservation(env)
        env = FrameStack(env, frame_stacking)

    if os.path.exists(model_name + ".zip"):
        model = RecurrentPPO.load(model_name, env=env, verbose=1,
                                  tensorboard_log=config.tensorboard_log_path)
        reset_timesteps = False
    else:
        model = RecurrentPPO("MlpPolicy", env=env, verbose=1,
                             tensorboard_log=config.tensorboard_log_path, n_steps=rollout_steps)
        reset_timesteps = True

    # model.learning_rate = 1.0
    # model.policy.optimizer = dadaptation.DAdaptAdam(model.policy.parameters(), lr=model.learning_rate)

    model.learn(total_timesteps=steps, log_interval=1, reset_num_timesteps=reset_timesteps,
                tb_log_name=model_name,
                callback=[
                    CheckpointCallback(save_freq=10000, save_path=config.checkpoint_path, name_prefix=model_name,
                                       save_replay_buffer=True, save_vecnormalize=True),
                    TrackmaniaInputCallback(), ProgressBarCallback()])
    model.save(model_name)
