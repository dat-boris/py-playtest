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

import gym.spaces as spaces

from .constant import Param
from .action import (
    ActionWaitRange,
    ActionRange,
    ActionFactory,
    ActionInstance,
    ActionWait,
    ActionBoolean,
    ActionBooleanRange,
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
    assert action.to_numpy_data().tolist() == [1]


class MockNewAction(ActionBoolean[MockState]):
    key = "new_action"


class MockNewActionRange(ActionBooleanRange[MockNewAction, MockState]):
    instance_class = MockNewAction


class MockActionFactory(ActionFactory):
    range_classes = [ActionWaitRange, MockNewActionRange]


@pytest.fixture
def factory():
    return MockActionFactory(Param(number_of_players=2))


def test_action_factory_action(factory):
    assert isinstance(factory.action_space, spaces.Dict)


def test_action_factory_possible(factory):
    possible_action_space = factory.action_space_possible
    assert isinstance(possible_action_space, spaces.Dict)

    assert spaces.flatdim(possible_action_space) == 2

    action_dict = factory.action_range_to_numpy(
        [ActionWaitRange(MockState(), player_id=0)]
    )
    assert action_dict["wait"] == [1]


def test_convert_action(factory):
    action_numpy = factory.to_numpy(MockNewAction())
    assert action_numpy.tolist() == [1, 0]
    action = factory.from_numpy(action_numpy)
    assert action == MockNewAction()
    action = factory.from_numpy(np.array([0, 1]))
    assert action == ActionWait()
