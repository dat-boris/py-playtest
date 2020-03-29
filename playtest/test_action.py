"""Action

This establish a case on how to test against Actions.

This design discuss the trade-off between noisy class design, vs.
a trade-off for accuracy.

e.g. The explicitness for two class, can we collapse into one class if okay?
Let's err on the side of noisiness for now.
"""
import re
import numpy as np
import pytest

from typing import Set, List

import gym.spaces as spaces

from .constant import Param
from .action import (
    ActionWaitRange,
    ActionRange,
    ActionFactory,
    ActionInstance,
    ActionWait,
    ActionSingleValue,
    ActionSingleValueRange,
    ActionValueInSet,
    ActionValueInSetRange,
    InvalidActionError,
)
from .test_state import MockState


def test_wait_range():
    action_range = ActionWaitRange(MockState(), player_id=0)
    action = ActionWait()
    assert action_range.is_valid(action)


def test_wait_numpy():
    action_range = ActionWaitRange(MockState(), player_id=0)
    assert action_range.to_numpy_data().tolist() == [1]
    action = ActionWait()
    assert action.to_int() == 0


class ActionBet(ActionSingleValue[MockState]):
    """A mock action for single value
    """

    key = "bet"

    minimum_value = 5
    maximum_value = 10

    def resolve(self, s, player_id, a=None):
        pass


class BetActionUpperLowerRange(ActionSingleValueRange[ActionBet, MockState]):
    """A mock action for single value, and use range method
    """

    instance_class = ActionBet

    def __init__(self, state, player_id):
        self.lower = 5
        self.upper = 9


def test_bet_action_single_value():
    action_range = BetActionUpperLowerRange(MockState(), player_id=0)
    assert str(action_range) == "bet(5->9)"
    action = ActionBet(5)
    assert action_range.is_valid(action)

    # Now check conversion
    assert action_range.get_action_space_possible() == spaces.Box(
        low=5, high=10, shape=(2,), dtype=np.int
    )
    assert action_range.to_numpy_data().tolist() == [5, 9]
    assert action_range.to_numpy_data_null().tolist() == [0, 0]

    assert action.to_int() == 5 - 5


class ActionEat(ActionValueInSet[MockState, str]):
    """A mock of action value in set"""

    key = "eat"

    value_set_mapping = ["apple", "orange", "banana"]
    unique_value_count = 3

    def resolve(self, s, player_id, a=None):
        pass


class EatActionSetRange(ActionValueInSetRange[ActionEat, MockState, str]):
    """An action range that takes a set, and gives a set"""

    instance_class = ActionEat
    possible_values: Set[str]

    def __init__(self, state, player_id):
        """A mock that assume all values are possible"""
        self.possible_values = {"apple", "banana"}


def test_action_in_set():
    action_range = EatActionSetRange(MockState(), player_id=0)
    # Get check on the action
    assert str(action_range) == "eat([apple,banana])"
    with pytest.raises(InvalidActionError):
        ActionEat("not_exists")
    action = ActionEat("banana")
    assert action_range.is_valid(action)

    # Now check the numpy conversion
    assert action_range.get_action_space_possible() == spaces.MultiBinary(3)
    assert action_range.to_numpy_data_null().tolist() == [0, 0, 0]
    assert action_range.to_numpy_data().tolist() == [1, 0, 1]

    assert action.to_int() == 2


# -----------------------
# Testing ActionFactory
# -----------------------


class MockActionFactory(ActionFactory):
    range_classes = [ActionWaitRange, BetActionUpperLowerRange, EatActionSetRange]


@pytest.fixture
def factory():
    return MockActionFactory(Param(number_of_players=2))


def test_action_factor_action_space(factory):
    assert factory.number_of_actions == 10


def test_action_factory_possible(factory):
    possible_action_space = factory.action_space_possible
    assert isinstance(possible_action_space, spaces.Dict)

    assert spaces.flatdim(possible_action_space) == 6

    action_dict = factory.action_range_to_numpy(
        [ActionWaitRange(MockState(), player_id=0)]
    )
    assert action_dict["wait"] == [1]


def test_convert_action(factory):
    action_map = factory.get_action_map()
    assert action_map == [
        (ActionWaitRange, 0, 1),
        (BetActionUpperLowerRange, 1, 7),
        (EatActionSetRange, 7, 10),
    ]
    action_value = factory.to_int(ActionEat("orange"))
    assert isinstance(action_value, int), "Convert action to np.int"
    assert action_value == 8
    action = factory.from_int(action_value)
    assert action == ActionEat("orange")
    action = factory.from_int(5)
    assert action == ActionBet(9)
    # since wait is the first item in MockActionFactory
    action = factory.from_int(0)
    assert action == ActionWait()
