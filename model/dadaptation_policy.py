from typing import Optional, Union, List, Dict, Type, Any

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.ppo import MultiInputPolicy
from torch import nn


class DAdaptationMultiInputPolicy(MultiInputPolicy):
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Space, lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh, ortho_init: bool = True, use_sde: bool = False,
                 log_std_init: float = 0.0, full_std: bool = True, use_expln: bool = False, squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None, share_features_extractor: bool = True,
                 normalize_images: bool = True, optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde,
                         log_std_init, full_std, use_expln, squash_output, features_extractor_class,
                         features_extractor_kwargs, share_features_extractor, normalize_images, optimizer_class,
                         optimizer_kwargs)