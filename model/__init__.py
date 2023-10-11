import gymnasium as gym
from model.trackmania_env import TrackmaniaEnvV1, TrackmaniaPreTrainingEnvV1
from model.trackmania_input_callback import TrackmaniaInputCallback

gym.register(id='Trackmania-v1', entry_point=TrackmaniaEnvV1)
gym.register(id='TrackmaniaPreTraining-v1', entry_point=TrackmaniaPreTrainingEnvV1)