from typing import List

import numpy as np
import gym.spaces as spaces

from .core import Component


class Token(Component):

    value: List[int]
    value_type = (int,)
    max_amount: int

    def __init__(self, value=None, max_amount=0xFFFF):
        if value is None:
            value = [0]
        self.max_amount = max_amount
        super().__init__(value)
        self.reset()

    @property
    def amount(self) -> int:
        return self.value[0]

    def reset(self):
        pass

    def take_from(self, other: "Token", value: int = None, all=True):
        if not value:
            assert all, "Must use all if no value specified"
            value = other.value[0]
        self.value[0] += value
        other.value[0] -= value

    def to_data(self):
        return self.value

    def get_observation_space(self):
        return spaces.Box(low=0, high=self.max_amount, shape=(1,), dtype=np.uint8)
