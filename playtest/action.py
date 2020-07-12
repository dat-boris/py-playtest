import re
import abc
import enum
import itertools
import random
from typing import (
    Optional,
    Type,
    Sequence,
    Dict,
    TypeVar,
    Generic,
    Set,
    List,
    Tuple,
    Union,
    MutableMapping,
    Any,
)
from dataclasses import dataclass

import numpy as np

import gym.spaces as spaces

from .state import FullState
from .logger import Announcer
from .constant import Param
from .components.core import Component

# For mapping typings
ActionEnum = enum.Enum


class InvalidActionError(RuntimeError):
    """Represent when we cannot Marshal this into a legal action.

    Note you should only throw this when action is illegal.
    e.g. it is a well formed action, but just not a legal move at the moment.

    Action is not well formed:
       e.g. Picking ActionBet of 11 when it is of range [0,10)
       - in this case, throw a normal python error

    Action is not legal.
       e.g. you only have Bank of 10 when you bet current action
       In this case you should throw InvalidActionError.
    """

    pass


@dataclass
class ActionInstance:
    """Represent an instance of the action.

    The instance of the action can be of acted on.
    """

    # Describe the name of the string
    key: ActionEnum
    value: int


class ActionRange(abc.ABC):
    """Represent a range of action

    The action range is responsible for representing a set of potential
    actions, and checking if an action is legal, based on the state
    provided for a specific player.

    Note that the range of action, might not be finite.  For example in a betting
    game, you can bet upwards to your bank amount, which could be infinite.
    """

    action_name: ActionEnum
    valid_range: Any

    def __init__(self, action_name: ActionEnum, valid_range: Any):
        self.action_name = action_name
        self.valid_range = valid_range

    @abc.abstractmethod
    def is_legal(self, x: ActionInstance, legal_range: Any) -> bool:
        """Check if action is valid"""
        raise NotImplementedError()

    @abc.abstractmethod
    def pick_random(self) -> ActionInstance:
        raise NotImplementedError()

    # ---------
    # Int marshalling - for openAI gym interaction
    # ---------

    def get_number_of_distinct_value(self) -> int:
        """Return the max possible value for the action

        Note this is inclusive.
        e.g. if action can take range of [0,1] -> max_value = 2
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def from_int(self, np_value: int) -> "ActionInstance":
        raise NotImplementedError()

    @abc.abstractmethod
    def to_int(self, value) -> int:
        raise NotImplementedError()

    # ---------
    # Str marshalling - for human interaction
    # ---------

    @abc.abstractmethod
    def from_str(self, action_str: str) -> "ActionInstance":
        raise NotImplementedError()

    # ---------
    # Gym space marshalling - for showing what action is available
    # ---------

    @abc.abstractmethod
    def get_action_space_possible(self) -> spaces.Space:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_numpy_data(self, legal_range) -> np.ndarray:
        """Return action space possible in numpy array
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to_numpy_empty_action(self) -> np.ndarray:
        """Return null data set for ActionRange
        """
        raise NotImplementedError()


class BaseDecision:
    """This base class for inheriting actions

    This class provides ability to:

    * Encode all action information in the game
    * Marshall / Un-marshall data into various format. In particular:
        - int format, for talking to OpenAI
        - str format, required for manual interaction with Human
        - action_space possibility, enumerating what is a current legal
          move for the actions
    * Ensure legality of the action based on decision given.
        - e.g. given a Set that can be of range [1,10), in current state of
          the game, maybe only a subset [1,5) is a legal move.  So we would
          like to execute that check.

    This is not meant to:

    * Specify the range against them.
    """

    # Constant
    action_enum: Type[enum.Enum]
    decision_ranges: MutableMapping[ActionEnum, ActionRange]

    # Specify default action for non-active player
    # TODO: remove if we do not need this
    # default: enum.Enum = enum.Enum.WAIT

    legal_action: Dict[ActionEnum, Any]

    def __init__(self, legal_action: Dict[ActionEnum, Any]):
        self.legal_action = legal_action

    @classmethod
    def get_number_of_actions(cls) -> int:
        """Represent the concret space for the action.

        This is used to communicate with OpenAI.gym about the the number of possible int.
        Specifying  the number of actions realted.

        See `action_space_possible` for explanation.
        """
        # We get this from the array of action
        return sum(
            [a.get_number_of_distinct_value() for a in cls.decision_ranges.values()]
        )

    @classmethod
    def action_space_possible(cls) -> spaces.Space:
        """This represent the observed possible action

        For example, for a Bet, you can bet a higher and lower amount, based
        on the bank of the player.

        Let's say a maximum bet range is between (0, 100)

        So:
        BetRange.action_space_possible == space.Box(2)
            # ^^^ The possible represent (lower, upper)

        # the output action space is always collapsed to int
        Bet.action_space == space.Box(1, lower, upper)
            # ^^^ Represent the one value, that can fall into above
        """
        return spaces.Dict(
            {
                action_key.name: action_range.get_action_space_possible()
                for action_key, action_range in cls.decision_ranges.items()
            }
        )

    def action_range_to_numpy(self) -> Dict[str, np.ndarray]:
        """Based on what action is legal, return a numpy space is avaliable.

        Return: a dict of recursive array which can be used for spaces.flatten
        """
        action_possible_dict = {}

        for action_key, action_range in self.decision_ranges.items():
            if action_key in self.legal_action:
                legal_value = self.legal_action[action_key]
                action_possible_dict[action_key.name] = action_range.to_numpy_data(
                    legal_value
                )
            else:
                action_possible_dict[
                    action_key.name
                ] = action_range.to_numpy_empty_action()

        return action_possible_dict

    def is_legal(self, action: ActionInstance) -> bool:
        action_enum_matched = action.key
        action_range = self.decision_ranges[action_enum_matched]
        legal_range = self.legal_action.get(action_enum_matched)
        # The action might not be in range at all
        if legal_range is None:
            return False
        return action_range.is_legal(action, legal_range)

    def pick_random_action(
        self, action_ranges: Optional[Sequence[ActionRange]] = None
    ) -> ActionInstance:
        """Pick a random action, out of the potential action classes"""
        if action_ranges is None:
            action_ranges = list(self.decision_ranges.values())
        action_range = random.choice(action_ranges)
        return action_range.pick_random()

    def from_str(self, action_input: str) -> ActionInstance:
        """Tokenize input from string into ActionInstance"""
        for action_key, action_range in self.decision_ranges.items():
            if action_input.startswith(action_key.value + "("):
                return action_range.from_str(action_input)
        raise KeyError(f"Unknown action: {action_input}")

    @classmethod
    def get_action_map(cls) -> Sequence[Tuple[enum.Enum, int, int]]:
        """return an array of action range and it's map

        For example, if we have too boolean action:

        The first two int range will belongs to the int
        [actionBoolARange, actionBoolARange, actionBoolBRange, actionBoolBRange]
        """
        action_map = []
        current_index = 0
        for action_enum, action_range in cls.decision_ranges.items():
            upper_bound = current_index + action_range.get_number_of_distinct_value()
            action_map.append((action_enum, current_index, upper_bound))
            current_index = upper_bound
        return action_map

    def to_int(self, action: ActionInstance) -> int:
        """Converting an action instance to numpy."""
        int_to_return = 0
        action_map = self.get_action_map()
        for action_enum, lower, upper in action_map:
            assert isinstance(action_enum, enum.Enum)
            if action_enum == action.key:
                action_range = self.decision_ranges[action_enum]
                value = action_range.to_int(action.value)
                final_action_value = lower + value
                assert lower <= final_action_value <= upper
                return final_action_value
        raise KeyError(f"Cannot map action: {action}")

    def from_int(self, input_value: int) -> ActionInstance:
        """Converting from numpy to an action instance."""
        int_space_searched = 0
        action_map = self.get_action_map()
        found_action = None
        for action_enum, lower, upper in action_map:
            assert isinstance(action_enum, enum.Enum)
            action_range = self.decision_ranges[action_enum]
            if lower <= input_value < upper:
                return action_range.from_int(input_value - lower)
        raise KeyError(f"Illegal action input: {input_value}.")

