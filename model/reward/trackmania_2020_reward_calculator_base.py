from input import Trackmania2020Data, TrackmaniaAction


class Trackmania2020Reward:
    def __init__(self, reward: float, terminated: bool, truncated: bool, info: dict):
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info


class Trackmania2020RewardCalculatorBase:
    def calculate_reward(self, action: TrackmaniaAction, trackmania_data: Trackmania2020Data) -> Trackmania2020Reward:
        raise NotImplementedError()

    def reset(self):
        pass
