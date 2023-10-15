from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class TrackmaniaInputCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        hparam_dict = {
            'algorithm': 'PPO',
            'learning_rate': self.model.learning_rate,
            'gamma': self.model.gamma,
        }

        metric_dict = {
            'rollout/ep_rew_mean': 0,
            'train/value_loss': 0.0,
        }

        self.logger.record('hparams', HParam(hparam_dict, metric_dict), exclude=('stdout', 'log', 'json', 'csv'))

    def _on_step(self) -> bool:
        info_items = self.locals.get('infos', [])

        if len(info_items) > 0:
            info: dict = info_items[0]

            is_accelerating = info.get('is_accelerating', False)
            is_braking = info.get('is_braking', False)
            is_turning_left = info.get('is_turning_left', False)
            is_turning_right = info.get('is_turning_right', False)
            speed = info.get('speed', 0)
            distance = info.get('distance', 0)
            gear = info.get('gear', 1)
            reward = info.get('reward', 0)
            max_reward = info.get('max_reward', 0)
            rpm = info.get('rpm', 0)
            average_speed = info.get('average_speed', 0)
            max_speed = info.get('max_speed', 0)
            max_rpm = info.get('max_rpm', 0)
            max_gear = info.get('max_gear', 0)

            self.logger.record('action/is_accelerating', is_accelerating)
            self.logger.record('action/is_braking', is_braking)
            self.logger.record('action/is_turning_left', is_turning_left)
            self.logger.record('action/is_turning_right', is_turning_right)
            self.logger.record('vehicle/speed', speed)
            self.logger.record('vehicle/gear', gear)
            self.logger.record('vehicle/rpm', rpm)
            self.logger.record('vehicle/average_speed', average_speed)
            self.logger.record('vehicle/max_speed', max_speed)
            self.logger.record('vehicle/max_rpm', max_rpm)
            self.logger.record('vehicle/max_gear', max_gear)
            self.logger.record('episode/distance', distance)
            self.logger.record('reward/reward', reward)
            self.logger.record('reward/max_reward', max_reward)

        return True

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

        self.training_env.reset()

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

        self.training_env.reset()

