import json
import os
import random
from datetime import datetime
from enum import Enum

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config
from input import TrackmaniaKeyboardDriver, TM2020OpenPlanetClient, TrackmaniaAction
from recording import ScreenRecorder
from utils.movement import movement


def discrete_action_to_enum(discrete_action: int) -> TrackmaniaAction:
    if discrete_action == 0:
        return TrackmaniaAction.Nothing
    elif discrete_action == 1:
        return TrackmaniaAction.Accelerate
    elif discrete_action == 2:
        return TrackmaniaAction.Brake
    elif discrete_action == 3:
        return TrackmaniaAction.Left
    elif discrete_action == 4:
        return TrackmaniaAction.Right
    elif discrete_action == 5:
        return TrackmaniaAction.AccelerateLeft
    elif discrete_action == 6:
        return TrackmaniaAction.AccelerateRight
    elif discrete_action == 7:
        return TrackmaniaAction.BrakeLeft
    elif discrete_action == 8:
        return TrackmaniaAction.BrakeRight
    elif discrete_action == 9:
        return TrackmaniaAction.DriftLeft
    elif discrete_action == 10:
        return TrackmaniaAction.DriftRight
    else:
        return TrackmaniaAction.Nothing


class TrackmaniaState(Enum):
    Speed = 1
    RPM = 2
    Distance = 3
    Terminated = 4


class TrackmaniaEnvV1(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(TrackmaniaEnvV1, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TrackmaniaAction))
        low_values = np.array([0] * 3 * 360 * 640, dtype=np.uint8)
        high_values = np.array([255] * 3 * 360 * 640, dtype=np.uint8)
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
        self._last_action: TrackmaniaAction = TrackmaniaAction.Nothing
        self._last_trackmania_data = None
        self._max_speed = 0
        self._max_rpm = 0
        self._max_gear = 0
        self._frame_number = 0
        self._total_speed = 0

        # Initial state
        self.state = None
        self.reset()

    def step(self, action):
        terminated = False
        truncated = False

        trackmania_action = discrete_action_to_enum(action)

        self._driver.perform_action(trackmania_action)

        last_speed = self.state['vector'][0]

        original_frame = self._screen_recorder.record_downsampled_frame(4)
        frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
        trackmania_data = self._openplanet_client.get_data()
        self.state = {
            'image': frame.flatten(),
            'vector': np.array([trackmania_data.speed, float(trackmania_data.rpm), float(trackmania_data.distance)],
                               dtype=np.float32)
        }

        speed_percentage_change = trackmania_data.speed / last_speed if last_speed > 0 else trackmania_data.speed

        # Define a simple reward structure
        reward = (trackmania_data.distance - self._previous_distance) * 10

        current_time = datetime.now()

        elapsed_time = current_time - self._start_time

        if self._last_action.has_flags(TrackmaniaAction.Accelerate,
                                         TrackmaniaAction.Brake) and self._last_action.has_xor_flags(
                                         TrackmaniaAction.Left, TrackmaniaAction.Right):
            reward *= 4
        elif self._last_action.has_flag(TrackmaniaAction.Accelerate) and not self._last_action.has_flag(TrackmaniaAction.Brake):
            reward *= 8
        elif self._last_action.has_flag(TrackmaniaAction.Brake):
            reward *= 0.5

        if (current_time - self._start_time).seconds < config.reward_timeframe_seconds:
            self._most_recent_states.append((trackmania_data.x, trackmania_data.y, trackmania_data.z))
        elif len(self._most_recent_states) > 0:
            self._most_recent_states.pop(0)
            self._most_recent_states.append((trackmania_data.x, trackmania_data.y, trackmania_data.z))

            reference_frame_distance = movement.calculate_distance(self._most_recent_states[0][0],
                                                                   self._most_recent_states[0][1],
                                                                   self._most_recent_states[0][2], trackmania_data.x,
                                                                   trackmania_data.y, trackmania_data.z)

            if reference_frame_distance < config.min_distance_traveled_threshold:
                reward = -1000

        if trackmania_data.speed < 5 and elapsed_time.total_seconds() > config.reward_timeframe_seconds * 4:
            reward = -10000
            truncated = True

        if speed_percentage_change < 0.85:
            reward = -1000

        if reward > 0:
            reward *= speed_percentage_change

        self._previous_distance = trackmania_data.distance

        if reward > self._max_reward:
            self._max_reward = reward

        # Check termination criteria
        if trackmania_data.terminated == 1:
            reward = self._max_reward * 10
            terminated = True
            self._driver.reset()
            self._driver.save_replay_and_start_course_again()

        if trackmania_data.distance < 50 and elapsed_time.total_seconds() > config.reward_timeframe_seconds * 4:
            truncated = True
            self._driver.reset()

        self._last_action = trackmania_action
        self._last_trackmania_data = trackmania_data

        if trackmania_data.speed > self._max_speed:
            self._max_speed = trackmania_data.speed
        if trackmania_data.rpm > self._max_rpm:
            self._max_rpm = trackmania_data.rpm
        if trackmania_data.gear > self._max_gear:
            self._max_gear = trackmania_data.gear
        self._total_speed += trackmania_data.speed
        self._frame_number += 1

        if self.render_mode == 'human':
            cv2.imshow('TrackmanAI', frame)

        return self.state, reward, terminated, truncated, {
            'is_accelerating': self._driver.current_action.has_flag(TrackmaniaAction.Accelerate),
            'is_braking': self._driver.current_action.has_flag(TrackmaniaAction.Brake),
            'is_turning_left': self._driver.current_action.has_flag(TrackmaniaAction.Left),
            'is_turning_right': self._driver.current_action.has_flag(TrackmaniaAction.Right),
            'max_speed': self._max_speed,
            'max_rpm': self._max_rpm,
            'max_gear': self._max_gear,
            'average_speed': self._total_speed / self._frame_number,
            'speed': trackmania_data.speed,
            'distance': trackmania_data.distance,
            'gear': trackmania_data.gear,
            'rpm': trackmania_data.rpm,
            'reward': reward,
            'max_reward': self._max_reward}

    def reset(self, **kwargs):
        # Reset environment state
        original_frame = self._screen_recorder.record_downsampled_frame(4)
        frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
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
        self._last_action = TrackmaniaAction.Nothing
        self._max_rpm = 0
        self._max_speed = 0
        self._total_speed = 0
        self._frame_number = 0
        return self.state, {}

    def render(self, mode='human'):
        # You can add visualization here. For now, just print the state.
        print(self.state['vector'])

    def close(self):
        pass


class TrackmaniaPreTrainingEnvV1(gym.Env):
    def __init__(self):
        super(TrackmaniaPreTrainingEnvV1, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(TrackmaniaAction))
        low_values = np.array([0] * 3 * 360 * 640, dtype=np.uint8)
        high_values = np.array([255] * 3 * 360 * 640, dtype=np.uint8)
        # self.observation_space = spaces.Box(low=low_values, high=high_values, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=low_values, high=high_values, dtype=np.uint8),
            'vector': spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                 high=np.array([350.0, 12000.0, np.inf], dtype=np.float32),
                                 dtype=np.float32)})
        self._max_reward = 0
        self._previous_distance = 0
        self._start_time = datetime.now()
        self._most_recent_states = []
        self._last_action = 0
        self._used_runs = []
        self._unused_runs = []
        self._current_run = None
        self._current_frame = 0
        self._frames = []

        # Initial state
        self.state = None
        self.reset()

    @staticmethod
    def _calculate_reward(action: int, accelerate: bool, brake: bool, steering_input: float):
        reward = 0

        if accelerate and brake:
            if action in [TrackmaniaActions.DriftLeft.value, TrackmaniaActions.DriftRight.value]:
                reward += 2
            else:
                reward -= 2
        elif accelerate:
            if action in [TrackmaniaActions.Accelerate.value, TrackmaniaActions.AccelerateLeft.value,
                          TrackmaniaActions.AccelerateRight.value]:
                reward += 2
            else:
                reward -= 2
        else:
            if action in [TrackmaniaActions.Brake.value, TrackmaniaActions.BrakeLeft.value,
                          TrackmaniaActions.BrakeRight.value]:
                reward += 2
            else:
                reward -= 2

        reward += TrackmaniaPreTrainingEnvV1._calculate_steering_reward(action, steering_input)

        return reward

    @staticmethod
    def _calculate_steering_reward(action: int, steering_input: float):
        if steering_input == 0 and action in [TrackmaniaActions.DriftLeft.value, TrackmaniaActions.DriftRight.value,
                                              TrackmaniaActions.AccelerateLeft.value,
                                              TrackmaniaActions.AccelerateRight.value,
                                              TrackmaniaActions.BrakeLeft.value,
                                              TrackmaniaActions.BrakeRight.value]:
            return -1
        elif steering_input == 0 and action in [TrackmaniaActions.Brake.value, TrackmaniaActions.Accelerate.value]:
            return 1
        elif steering_input > 0:
            if action in [TrackmaniaActions.DriftLeft.value, TrackmaniaActions.AccelerateLeft.value,
                          TrackmaniaActions.BrakeLeft.value]:
                return 1
            else:
                return -1
        elif steering_input < 0:
            if action in [TrackmaniaActions.DriftRight.value, TrackmaniaActions.AccelerateRight.value,
                          TrackmaniaActions.BrakeRight.value]:
                return 1
            else:
                return -1
        return 0

    def step(self, action):
        terminated = False
        truncated = False

        steering_input = round(self._frames[self._current_frame].steering_input)
        accelerate = bool(self._frames[self._current_frame].accelerate)
        brake = bool(self._frames[self._current_frame].brake)

        reward = TrackmaniaPreTrainingEnvV1._calculate_reward(action, accelerate, brake, steering_input)

        if reward > 0:
            reward *= self._frames[self._current_frame].speed / 10.0

        self._last_action = action
        self._current_frame += 1

        if self._current_frame >= len(self._frames):
            return self.state, reward, True, False, {}

        self.state = {
            'image': self._frames[self._current_frame].frame.flatten(),
            'vector': np.array([self._frames[self._current_frame].speed, self._frames[self._current_frame].rpm,
                                self._frames[self._current_frame].distance],
                               dtype=np.float32)
        }

        if self.render_mode == 'human':
            cv2.imshow('Trackmania', self._frames[self._current_frame].frame)

        return self.state, reward, terminated, truncated, {'max_reward': self._max_reward}

    def reset(self, **kwargs):
        # Reset environment state
        self._start_time = datetime.now()
        self._max_reward = 0
        self._last_action = TrackmaniaAction.Nothing.value
        self._used_runs = []
        self._unused_runs = []
        self._current_run = None
        self._current_frame = 0

        if len(self._unused_runs) == 0 and len(self._used_runs) == 0:
            self._unused_runs = os.listdir(config.data_base_path)
            random.shuffle(self._unused_runs)

        if self._current_run is not None:
            self._used_runs.append(self._current_run)

        self._current_run = self._unused_runs.pop(0)

        loaded_run = self._load_frames_for_run(self._current_run)

        while not loaded_run:
            if self._current_run is not None:
                self._used_runs.append(self._current_run)

            if len(self._unused_runs) == 0 and len(self._used_runs) > 0:
                self._unused_runs = self._used_runs
                random.shuffle(self._unused_runs)
                self._used_runs = []

            self._current_run = self._unused_runs.pop(0)

            loaded_run = self._load_frames_for_run(self._current_run)

        self.state = {
            'image': self._frames[0].frame.flatten(),
            'vector': np.array([self._frames[0].speed, self._frames[0].rpm,
                                self._frames[0].distance],
                               dtype=np.float32)
        }
        return self.state, {}

    def _load_frames_for_run(self, run):
        self._current_frame = 0

        for idx in range(int('9' * config.frame_naming_places)):
            frame_name = (config.frame_naming_places - len(str(idx))) * '0' + str(idx)

            frame_path = os.path.join(config.data_base_path, run, f'{frame_name}.bmp')
            json_path = os.path.join(config.data_base_path, run, f'{frame_name}.json')

            if not os.path.exists(frame_path) or not os.path.exists(json_path):
                break

            with open(json_path, 'r') as json_file:
                frame_data = json.load(json_file)

                self._frames.append(PreTrainingFrameData(frame_data, cv2.imread(frame_path)))

        return len(self._frames) > 0

    def render(self, mode='human'):
        if mode == 'human':
            self.render_mode = 'human'
            cv2.namedWindow('Trackmania', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Trackmania', 640, 360)

        if mode == 'rgb_array':
            return self.state['image']

    def close(self):
        pass


class PreTrainingFrameData:
    def __init__(self, frame_data: dict, frame: np.ndarray):
        self.frame: np.ndarray = frame
        self.x: float = frame_data['x']
        self.y: float = frame_data['y']
        self.z: float = frame_data['z']
        self.speed: float = frame_data.get('speed', frame_data.get('unknown_0', 0.0))
        self.distance: float = frame_data.get('distance', frame_data.get('unknown_1', 0.0))
        self.gear: int = frame_data.get('gear', frame_data.get('unknown_5', 0.0))
        self.accelerate: float = frame_data['accelerate']
        self.brake: float = frame_data['brake']
        self.steering_input = frame_data['steering_input']
        self.rpm: float = frame_data['rpm']
