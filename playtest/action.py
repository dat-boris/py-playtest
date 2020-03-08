import re
from typing import Optional, Type, Sequence, Dict

import numpy as np

import gym.spaces as spaces

from .state import FullState
from .logger import Announcer
from .utils import Param
from .components.core import Component


class InvalidActionError(RuntimeError):
    pass


class ActionInstance:
    """Represent an instance of the action
    """

    key: str

    # possible action space for the action
    action_space = spaces.MultiBinary(1)

    def __init__(self):
        # Only need overriding if there's parameter
        pass

    def __eq__(self, x):
        return isinstance(x, self.__class__) and self.key == x.key

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        """Create string representation"""
        return self.key

    @classmethod
    def from_str(cls, action_str: str) -> "ActionInstance":
        if action_str == cls.key:
            return cls()
        raise InvalidActionError(f"Unknown action: {action_str}")

    def to_numpy_data(self) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([0])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "ActionInstance":
        """Check if value is acceptable"""
        assert len(array) == 1
        if array[0] == 1:
            return cls()
        raise ValueError(f"Unknown value {array} for {cls}")

    def resolve(self, s: FullState, player_id: int, a: Optional[Announcer] = None):
        raise NotImplementedError()


class ActionRange:
    """Represent a range of action"""

    instance_class: Type[ActionInstance]

    action_space_possible = spaces.MultiBinary(1)

    def __init__(self):
        pass

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.instance_class.key}"

    def __eq__(self, x):
        return self.__class__ == x.__class__

    def to_numpy_data(self) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([0])

    def is_valid(self, x: ActionInstance) -> bool:
        return isinstance(x, self.instance_class)


class ActionWait(ActionInstance):
    key = "wait"


class ActionWaitRange(ActionRange):
    instance_class = ActionWait


# Sentinel Action instance for comparison
ACTION_WAIT = ActionWait()


class ActionSingleValue(ActionInstance):
    """A base class for single value action"""

    value: int

    # possible action space for the action
    # Fill in here for subclass. For example:
    # action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int8)
    action_space: spaces.Box

    def __init__(self, value: str):
        self.value = int(value)

    def __repr__(self):
        return f"{self.key}({self.value})"

    def __eq__(self, x):
        return self.key == x.key and self.value == x.value

    @classmethod
    def from_str(cls, action_str: str) -> ActionInstance:
        action_key = cls.key
        matches = re.match(f"{action_key}[(](\\d+)[)]", action_str)
        if matches:
            return cls(matches.group(1))
        raise InvalidActionError(f"Unknown action: {action_str}")

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.value])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([0])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> ActionInstance:
        """Check if value is acceptable"""
        assert len(array) == 1
        assert array[0] > 0, "Must provide positive bet value"
        return cls(array[0])


class ActionSingleValueRange(ActionRange):
    # Fill in instanceClass here
    # e.g. instance_class = ActionBet
    upper: int
    lower: int

    min_lower: int
    max_upper: int

    # Fill in action_class_possible  here
    # action_space_possible = spaces.Box(
    #     low=0, high=100, shape=(2,), dtype=np.int8)

    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower

    def __repr__(self):
        return f"{self.instance_class.key}({self.lower}->{self.upper})"

    def __eq__(self, x):
        return (
            self.__class__ == x.__class__
            and self.upper == x.upper
            and self.lower == x.lower
        )

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.lower, self.upper])

    @classmethod
    def to_numpy_data_null(self) -> np.ndarray:
        return np.array([self.min_lower, self.max_upper])

    def is_valid(self, x) -> bool:
        if isinstance(x, self.instance_class):
            return self.lower <= x.value <= self.upper  # type: ignore
        return False


class ActionFactory:

    param: Param
    range_classes: Sequence[Type[ActionRange]]
    # Specify default action for non-active player
    default: ActionInstance = ActionWait()

    def __init__(self, param: Param):
        self.param = param

    @property
    def action_space(self) -> spaces.Space:
        """Represent the concret space for the action

        See `action_space_possible` for explanation.
        """
        # We get this from the array of action
        action_space_dict: Dict[str, spaces.Space] = {}
        for a in self.range_classes:
            action_key = a.instance_class.key
            # Note that instance class is an object, we need to work with this
            action_space = a.instance_class.action_space
            assert isinstance(
                action_space, spaces.Space
            ), f"{action_key} does not have valid action space"
            action_space_dict[action_key] = action_space
        return spaces.Dict(action_space_dict)

    @property
    def action_space_possible(self) -> spaces.Space:
        """This represent the observed possible action

        For example, for a Bet, you can bet a higher and lower amount, based
        on the bank of the player.

        Let's say a maximum bet range is between (0, 100)

        So:
        Bet.action_space_possible == space.Box(2)
            # ^^^ The possible represent 2 values

        Bet.to_numpy_data_null == [0,100]
            # ^^^^ The value of default action

        Bet.action_space == space.Box(1)
            # ^^^ Represent the one value, that can fall into above

        """
        return spaces.Dict(
            {a.instance_class.key: a.action_space_possible for a in self.range_classes}
        )

    def action_range_to_numpy(
        self, action_possibles: Sequence[ActionRange]
    ) -> Dict[str, np.ndarray]:
        """Based on the list of Action Range, return a list of action possible

        Return: a list of recursive array which can be used for spaces.flatten
        """
        action_possible_dict = {
            a.instance_class.key: a.to_numpy_data_null() for a in self.range_classes
        }
        for a in action_possibles:
            action_key = a.instance_class.key
            assert action_key in action_possible_dict, "Unknown action dict!"
            action_possible_dict[action_key] = a.to_numpy_data()

        return action_possible_dict

    def from_str(self, action_input: str) -> ActionInstance:
        """Tokenize input from string into ActionInstance"""
        for a in self.range_classes:
            try:
                return a.instance_class.from_str(action_input)
            except InvalidActionError:
                pass
        raise InvalidActionError(f"Unknown action: {action_input}")

    def to_numpy(self, action: ActionInstance) -> np.ndarray:
        action_dict = {
            a.instance_class.key: a.instance_class.to_numpy_data_null()
            for a in self.range_classes
        }
        for action_range in self.range_classes:
            action_expected = action_range.instance_class
            if action.__class__ is action_expected:
                action_dict[action_expected.key] = action.to_numpy_data()
        return spaces.flatten(self.action_space, action_dict)

    def from_numpy(self, numpy_input: np.ndarray) -> ActionInstance:
        unflattened = spaces.unflatten(self.action_space, numpy_input)
        # Now given the dict, check if any of them are engaged
        err_msgs = []
        for a in self.range_classes:
            numpy_val = unflattened[a.instance_class.key]
            try:
                return a.instance_class.from_numpy(numpy_val)
            except (ValueError, AssertionError) as e:
                err_msgs.append(str(e))
        raise InvalidActionError(
            f"Invalid action input: {numpy_input}.  Msg: {err_msgs}"
        )
