from datetime import datetime
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from input import TrackmaniaKeyboardDriver, TM2020OpenPlanetClient
from recording import ScreenRecorder


class TrackmaniaActions(Enum):
    Nothing = 0
    Accelerate = 1
    Brake = 2
    Left = 3
    Right = 4
    AccelerateLeft = 5
    AccelerateRight = 6
    BrakeLeft = 7
    BrakeRight = 8
    DriftLeft = 9
    DriftRight = 10


class TrackmaniaState(Enum):
    Speed = 1
    RPM = 2
    Distance = 3
    Terminated = 4


class TrackmaniaEnvV1(gym.Env):
    def __init__(self):
        super(TrackmaniaEnvV1, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TrackmaniaActions))
        low_values = np.array([0] * 4 * 360 * 640, dtype=np.uint8)
        high_values = np.array([255] * 4 * 360 * 640, dtype=np.uint8)
        # self.observation_space = spaces.Box(low=low_values, high=high_values, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=low_values, high=high_values, dtype=np.uint8),
            'vector': spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                 high=np.array([350.0, 12000.0, np.inf], dtype=np.float32),
                                 dtype=np.float32)})
        self._driver = TrackmaniaKeyboardDriver()
        self._max_reward = 0
        self._previous_distance = 0
        self._screen_recorder = ScreenRecorder()
        self._openplanet_client = TM2020OpenPlanetClient()
        self._start_time = datetime.now()

        # Initial state
        self.state = None
        self.reset()

    def step(self, action):
        # Define your action dynamics here       # switch over actions
        if action == TrackmaniaActions.Nothing.value:
            self._driver.reset()
        elif action == TrackmaniaActions.Accelerate.value:
            self._driver.accelerate()
        elif action == TrackmaniaActions.Brake.value:
            self._driver.brake()
        elif action == TrackmaniaActions.Left.value:
            self._driver.turn_left()
        elif action == TrackmaniaActions.Right.value:
            self._driver.turn_right()
        elif action == TrackmaniaActions.AccelerateLeft.value:
            self._driver.accelerate_left()
        elif action == TrackmaniaActions.AccelerateRight.value:
            self._driver.accelerate_right()
        elif action == TrackmaniaActions.BrakeLeft.value:
            self._driver.brake_left()
        elif action == TrackmaniaActions.BrakeRight.value:
            self._driver.brake_right()
        elif action == TrackmaniaActions.DriftLeft.value:
            self._driver.drift_left()
        elif action == TrackmaniaActions.DriftRight.value:
            self._driver.drift_right()

        frame = self._screen_recorder.record_downsampled_frame(4)
        trackmania_data = self._openplanet_client.get_data()
        self.state = {
            'image': frame.flatten(),
            'vector': np.array([trackmania_data.speed, float(trackmania_data.rpm), float(trackmania_data.distance)],
                               dtype=np.float32)
        }

        # Define a simple reward structure
        reward = trackmania_data.distance - self._previous_distance
        self._previous_distance = trackmania_data.distance

        if reward > self._max_reward:
            self._max_reward = reward

        # Check termination criteria
        terminated = trackmania_data.terminated == 1
        truncated = False

        current_time = datetime.now()

        elapsed_time = current_time - self._start_time

        if trackmania_data.distance < 1000 and elapsed_time.seconds > 10:
            truncated = True

        return self.state, reward, terminated, truncated, {'is_a_pressed': self._driver.is_a_pressed,
                                                           'is_d_pressed': self._driver.is_d_pressed,
                                                           'is_w_pressed': self._driver.is_w_pressed,
                                                           'is_s_pressed': self._driver.is_s_pressed,
                                                           'max_reward': self._max_reward}

    def reset(self, **kwargs):
        # Reset environment statews
        frame = self._screen_recorder.record_downsampled_frame(4)
        trackmania_data = self._openplanet_client.get_data()
        self.state = {
            'image': frame.flatten(),
            'vector': np.array([trackmania_data.speed, float(trackmania_data.rpm), float(trackmania_data.distance)],
                               dtype=np.float32)
        }
        self._driver.reset()
        self._driver.restart()
        self._start_time = datetime.now()
        self._max_reward = 0
        return self.state, {}

    def render(self, mode='human'):
        # You can add visualization here. For now, just print the state.
        print(self.state[1:])

    def close(self):
        pass
