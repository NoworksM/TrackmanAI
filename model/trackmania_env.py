from datetime import datetime
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from input import TrackmaniaKeyboardDriver, TM2020OpenPlanetClient
from recording import ScreenRecorder
from utils.movement import movement


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


FORWARD_ACTIONS = [TrackmaniaActions.Accelerate.value, TrackmaniaActions.AccelerateLeft.value,
                   TrackmaniaActions.AccelerateRight.value]

DRIFT_ACTIONS = [TrackmaniaActions.DriftLeft.value, TrackmaniaActions.DriftRight.value]

BRAKE_ACTIONS = [TrackmaniaActions.Brake.value, TrackmaniaActions.BrakeLeft.value, TrackmaniaActions.BrakeRight.value]


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
        self._most_recent_states = []
        self._last_action = 0

        # Initial state
        self.state = None
        self.reset()

    def step(self, action):
        terminated = False
        truncated = False

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
        reward = (trackmania_data.distance - self._previous_distance) * 10

        current_time = datetime.now()

        elapsed_time = current_time - self._start_time

        if self._last_action in FORWARD_ACTIONS:
            reward *= 8
        elif self._last_action in DRIFT_ACTIONS:
            reward *= 4
        elif self._last_action in BRAKE_ACTIONS:
            reward *= 0.5

        if (current_time - self._start_time).seconds < config.reward_timeframe_seconds:
            self._most_recent_states.append((trackmania_data.x, trackmania_data.y, trackmania_data.z))
        else:
            self._most_recent_states.pop(0)
            self._most_recent_states.append((trackmania_data.x, trackmania_data.y, trackmania_data.z))

            reference_frame_distance = movement.calculate_distance(self._most_recent_states[0][0],
                                                                   self._most_recent_states[0][1],
                                                                   self._most_recent_states[0][2], trackmania_data.x,
                                                                   trackmania_data.y, trackmania_data.z)

            if reference_frame_distance < config.min_distance_traveled_threshold:
                reward = -100

        if trackmania_data.speed < 5 and elapsed_time.total_seconds() > config.reward_timeframe_seconds * 4:
            reward = -100
            truncated = True

        self._previous_distance = trackmania_data.distance

        if reward > self._max_reward:
            self._max_reward = reward

        # Check termination criteria
        if trackmania_data.terminated == 1:
            terminated = True

        if trackmania_data.distance < 50 and elapsed_time.total_seconds() > config.reward_timeframe_seconds * 4:
            truncated = True

        self._last_action = action

        return self.state, reward, terminated, truncated, {'is_a_pressed': self._driver.is_a_pressed,
                                                           'is_d_pressed': self._driver.is_d_pressed,
                                                           'is_w_pressed': self._driver.is_w_pressed,
                                                           'is_s_pressed': self._driver.is_s_pressed,
                                                           'max_reward': self._max_reward}

    def reset(self, **kwargs):
        # Reset environment state
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
        self._last_action = TrackmaniaActions.Nothing.value
        return self.state, {}

    def render(self, mode='human'):
        # You can add visualization here. For now, just print the state.
        print(self.state[1:])

    def close(self):
        pass
