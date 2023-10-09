import os

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import PPO
import gymnasium as gym

from model.trackmania_env import TrackmaniaEnvV1

gym.register(id='Trackmania-v1', entry_point=TrackmaniaEnvV1)

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Assume 3-channel RGB image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        # Calculate the resultant shape after CNN layers
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space["image"].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 2, features_dim),  # +2 for speed and RPM
            nn.ReLU()
        )

    def forward(self, observations):
        # Split the observations. Assume images are under 'image' key
        # and other features are flattened into a single vector.
        image_obs = observations["image"]
        scalar_obs = observations["vector"]

        image_embeddings = self.cnn(image_obs)
        # Concatenate image embeddings with scalar observations
        concatenated_features = th.cat([image_embeddings, scalar_obs], dim=1)
        return self.linear(concatenated_features)


policy_kwargs = {
    "features_extractor_class": CustomCNN,
    "features_extractor_kwargs": {"features_dim": 128},
    "activation_fn": th.nn.ReLU,
    "net_arch": [128, dict(pi=[64, 32], vf=[64, 32])],  # Adjust as needed
}


if os.path.exists("trackmanai_ppo.zip"):
    model = PPO.load("trackmanai_ppo", env=gym.make('Trackmania-v1'), verbose=1)
else:
    model = PPO("MultiInputPolicy", 'Trackmania-v1', verbose=1)

while True:
    model.learn(total_timesteps=10000)
    model.save("trackmanai_ppo")
