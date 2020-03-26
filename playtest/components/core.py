import abc
import numpy as np
from typing import Dict, Type, List, Sequence

import gym.spaces as spaces


class Component(abc.ABC):
    """Core component class that is to be inherited
    """

    def __init__(self, data, param=None):
        """Initialize state.

        :param param: Decide if we are going to initialize with param
        """
        pass

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def to_data(self):
        raise NotImplementedError()

    @classmethod
    def from_data(cls, data):
        return cls(data)

    @abc.abstractmethod
    def to_numpy_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_observation_space(self) -> spaces.Space:
        raise NotImplementedError()

    @property
    def observation_space(self) -> spaces.Space:
        return self.get_observation_space()
