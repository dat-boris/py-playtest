"""Action

This establish a case on how to test against Actions.

This design discuss the trade-off between noisy class design, vs.
a trade-off for accuracy.

e.g. The explicitness for two class, can we collapse into one class if okay?
Let's err on the side of noisiness for now.
"""
import re
import enum
import numpy as np
from collections import OrderedDict
from typing import Set, List, Union

import pytest

import gym.spaces as spaces

from .constant import Param
import playtest.action as acn
from .test_state import MockState
from .action_range import action_bool, action_int


class MockActionName(enum.Enum):
    DECIDE_BOOLEAN = "bool"
    DECIDE_INT_IN_RANGE = "range"
    DECIDE_INT_IN_SET = "set"


class MockDecision(acn.BaseDecision):
    """Mock decision for testing rendering
    """

    action_enum = MockActionName

    decision_ranges = OrderedDict(
        [
            (
                MockActionName.DECIDE_BOOLEAN,
                action_bool.ActionBooleanRange(MockActionName.DECIDE_BOOLEAN, None),
            ),
            (
                MockActionName.DECIDE_INT_IN_SET,
                action_int.ActionIntInSet(MockActionName.DECIDE_INT_IN_SET, {1, 3, 5}),
            ),
            (
                MockActionName.DECIDE_INT_IN_RANGE,
                action_int.ActionIntInRange(
                    MockActionName.DECIDE_INT_IN_RANGE, (10, 20)
                ),
            ),
        ]
    )


@pytest.fixture
def md() -> MockDecision:
    md = MockDecision(
        # This gives the the specific decisions space
        {
            MockActionName.DECIDE_BOOLEAN: True,
            # Note this is subset of the set
            MockActionName.DECIDE_INT_IN_SET: {3, 5},
            MockActionName.DECIDE_INT_IN_RANGE: (11, 13),
        }
    )
    return md


# ----------------------
# Testing util functions
# ----------------------


def test_action_factor_action_space(md: MockDecision):
    assert md.number_of_actions == 14


def test_pick_random_action(md: MockDecision):
    random_action = md.pick_random_action()
    assert isinstance(random_action, acn.ActionInstance)


# ----------------------
# Testing boolean mapping
# ----------------------


def test_decision_check_bool(md: MockDecision):
    """Test that we created out decisions correctly
    """
    bool_action = md.from_str("bool()")
    assert md.is_legal(bool_action)


def test_md_to_action_int_bool(md: MockDecision):
    """Action only takes int, so we have to convert this
    into a valid int somehow!
    """
    action_map = md.get_action_map()
    assert action_map == [
        (MockActionName.DECIDE_BOOLEAN, 0, 1),
        (MockActionName.DECIDE_INT_IN_SET, 1, 4),
        (MockActionName.DECIDE_INT_IN_RANGE, 4, 14),
    ]

    # TODO: this is probably not needed
    # action_value = md.to_int({MockActionName.DECIDE_BOOLEAN: True})
    # assert isinstance(action_value, int), "Convert action to np.int"
    # assert action_value == 1

    expected_action = acn.ActionInstance(MockActionName.DECIDE_BOOLEAN, True)
    action = md.from_int(0)
    assert isinstance(action, acn.ActionInstance)
    assert action == expected_action

    expected_action = acn.ActionInstance(MockActionName.DECIDE_INT_IN_SET, 3)
    # Lower value (1) + offset (3 -> 1)
    action = md.from_int(2)
    assert isinstance(action, acn.ActionInstance)
    assert action == expected_action

    expected_action = acn.ActionInstance(MockActionName.DECIDE_INT_IN_RANGE, 11)
    # Lower value (4) + offset (1)
    action = md.from_int(5)
    assert isinstance(action, acn.ActionInstance)
    assert action == expected_action


def test_set_action(md: MockDecision):
    expected_action = acn.ActionInstance(MockActionName.DECIDE_INT_IN_SET, 1)
    int_in_set_action = md.from_str("set(3)")
    assert md.is_legal(int_in_set_action)

    invalid_action = md.from_str("set(10)")
    assert not md.is_legal(invalid_action)


def test_range_action(md: MockDecision):
    int_in_range_action = md.from_str("range(11)")
    assert md.is_legal(int_in_range_action)

    invalid_action = md.from_str("range(17)")
    assert not md.is_legal(invalid_action)
