from typing import List

import numpy as np
import gym.spaces as spaces

from .core import Component


class Counter(Component):

    value: List[int]
    value_type = (int,)
    max_amount = 0xFFFF

    def increment(self, value=1):
        self.value[0] += value

    @classmethod
    def get_observation_space(cls):
        return spaces.Box(low=0, high=cls.max_amount, shape=(1,), dtype=np.uint8)
