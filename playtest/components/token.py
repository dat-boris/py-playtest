import numpy as np
import gym.spaces as spaces

from .core import Component


class Token(Component):

    amount: int
    init_amount: int
    max_amount: int

    def __str__(self):
        return str(self.amount)

    def __init__(self, amount=0, max_amount=0xFFFF):
        self.max_amount = max_amount
        self.init_amount = amount
        self.reset()

    def reset(self):
        self.amount = self.init_amount

    def take_from(self, other: "Token", value=None, all=True):
        if not value:
            assert all, "Must use all if no value specified"
            value = other.amount
        self.amount += value
        other.amount -= value

    def to_data(self):
        return self.amount

    def get_observation_space(self):
        return spaces.Box(low=0, high=self.max_amount, shape=(1,), dtype=np.uint8)

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.amount], dtype=np.uint8)
