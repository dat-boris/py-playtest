import sys
import abc
import enum
import ast
import inspect
import numpy as np
import typing
from typing import Dict, Type, List, Sequence, Tuple, Union

import typeguard

import gym.spaces as spaces

# A list of seperator that can be used to separate elements
SEPERATOR = ","


class Component(abc.ABC):
    """Core component class that is to be inherited
    """

    # Note this is a tuple - since this maps to the
    # open_ai_gym.Box space
    value: Union[List]
    # TODO: remove this for python3.8, this is the type reflection of above
    value_type: Tuple[Type[Union[enum.IntEnum, int]]]

    def __init__(self, value: List, param=None):
        """Initialize state.

        :param param: Decide if we are going to initialize with param
        """
        typeguard.check_argument_types()
        self.value = value

    def __eq__(self, x):
        """Return equality if structure is deeply equal"""
        # Numpy supports deep comparison n
        return (self.to_numpy_data() == x.to_numpy_data()).all()

    def __repr__(self):
        """Return a readable string with seperator seperating"""
        return SEPERATOR.join(
            [d.name if isinstance(d, enum.IntEnum) else str(d) for d in self.value]
        )

    @classmethod
    def __get_value_type(cls) -> List[Type[Union[enum.IntEnum, int]]]:
        """Inspect the type signature of value"""
        if sys.version_info < (3, 8):
            assert cls.__get_value_type, "Must define value_type in python < ver3.8"
            # TODO: ignore Pre 3.8 fixes
            return cls.value_type  # type: ignore
        sig = inspect.signature(cls)
        value_type = sig.parameters["value"]
        return typing.get_args(value_type)

    @classmethod
    def from_str(cls, s):
        """Return the object from parsing the input string

        This automatically convert various enum into the properties.
        """
        cls_value_type = cls.__get_value_type()
        return cls(
            value=[
                cls_value_type[i][sv]
                if issubclass(cls_value_type[i], enum.IntEnum)
                else int(sv)
                for i, sv in enumerate(s.split(SEPERATOR))
            ]
        )

    def to_data(self) -> List[int]:
        """Return a list of integer to be represented
        as data
        """
        return [int(v) for v in self.value]

    @classmethod
    def from_data(cls, data):
        cls_value_type = cls.__get_value_type()
        data_value = []
        if data == cls.get_null_data():
            # null data, skipping
            return None
        for i, sv in enumerate(data):
            sv_type = cls_value_type[i]
            if issubclass(sv_type, Component):
                data_value.append(sv_type.from_data(sv))
            else:
                data_value.append(sv_type(sv))
        return cls(data_value)

    def to_numpy_data(self):
        return spaces.flatten(self.get_observation_space(), self.to_data())

    @classmethod
    def get_null_data(cls):
        return [0 for _ in cls.value_type]

    @classmethod
    @abc.abstractmethod
    def get_observation_space(cls) -> spaces.Space:
        raise NotImplementedError()
