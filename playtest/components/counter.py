import numpy as np
import gym.spaces as spaces

from .core import Component


class Counter(Component):

    value: int
    max_amount = 0xFFFF

    def __str__(self):
        return str(self.value)

    def __init__(self, value=None):
        self.reset()
        if value is not None:
            self.value = value

    def increment(self, value=1):
        self.value += value

    def reset(self):
        self.value = 0

    def to_data(self):
        return self.value

    def get_observation_space(self):
        return spaces.Box(low=0, high=self.max_amount, shape=(1,), dtype=np.uint8)

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.value], dtype=np.uint8)
