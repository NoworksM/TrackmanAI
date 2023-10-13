from datetime import datetime

import config
from input import Trackmania2020Data, TrackmaniaAction
from model.reward.trackmania_2020_reward_calculator_base import Trackmania2020RewardCalculatorBase, Trackmania2020Reward
from utils.movement import movement


class TrackmaniaRewardCalculatorRace(Trackmania2020RewardCalculatorBase):
    def __init__(self):
        self._max_speed = 0
        self._last_action = TrackmaniaAction.Nothing
        self._last_data = Trackmania2020Data((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        self._previous_distance = 0
        self._start_time = datetime.now()
        self._most_recent_states = []
        self._max_reward = 0
        self._max_rpm = 0
        self._max_gear = 0
        self._frame_number = 0
        self._total_speed = 0

    def calculate_reward(self, action: TrackmaniaAction, trackmania_data: Trackmania2020Data) -> Trackmania2020Reward:
        truncated = False
        terminated = False

        speed_percentage_change = trackmania_data.speed / self._last_data.speed if self._last_data.speed > 0 else 1

        # Define a simple reward structure
        reward = (trackmania_data.distance - self._previous_distance) * 10

        current_time = datetime.now()

        elapsed_time = current_time - self._start_time

        if self._last_action.has_flags(TrackmaniaAction.Accelerate,
                                       TrackmaniaAction.Brake) and self._last_action.has_xor_flags(
                                       TrackmaniaAction.Left, TrackmaniaAction.Right):
            reward *= 4
        elif self._last_action.has_flag(TrackmaniaAction.Accelerate) and not self._last_action.has_flag(
                TrackmaniaAction.Brake):
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
            reward = (-10000 * (1 - speed_percentage_change))

        if reward > 0:
            reward *= speed_percentage_change

            if trackmania_data.speed > 25:
                reward *= 2

        self._previous_distance = trackmania_data.distance

        if reward > self._max_reward:
            self._max_reward = reward

        # Check termination criteria
        if trackmania_data.terminated == 1:
            reward = self._max_reward * 10
            terminated = True

        if trackmania_data.distance < 50 and elapsed_time.total_seconds() > config.reward_timeframe_seconds * 4:
            truncated = True

        self._last_action = action
        self._last_data = trackmania_data

        if trackmania_data.speed > self._max_speed:
            self._max_speed = trackmania_data.speed
        if trackmania_data.rpm > self._max_rpm:
            self._max_rpm = trackmania_data.rpm
        if trackmania_data.gear > self._max_gear:
            self._max_gear = trackmania_data.gear
        self._total_speed += trackmania_data.speed
        self._frame_number += 1

        return Trackmania2020Reward(reward, truncated, terminated, {
            'max_speed': self._max_speed,
            'max_rpm': self._max_rpm,
            'max_gear': self._max_gear,
            'average_speed': self._total_speed / self._frame_number,
            'speed': trackmania_data.speed,
            'distance': trackmania_data.distance,
            'gear': trackmania_data.gear,
            'rpm': trackmania_data.rpm,
            'reward': reward,
            'max_reward': self._max_reward
        })

    def reset(self):
        self._last_data = Trackmania2020Data((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        self._last_action = TrackmaniaAction.Nothing
        self._max_speed = 0
        self._previous_distance = 0
        self._start_time = datetime.now()
        self._most_recent_states = []
        self._max_reward = 0
        self._max_rpm = 0
        self._max_gear = 0
        self._frame_number = 0
        self._total_speed = 0