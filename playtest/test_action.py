"""Action

This establish a case on how to test against Actions.

This design discuss the trade-off between noisy class design, vs.
a trade-off for accuracy.

e.g. The explicitness for two class, can we collapse into one class if okay?
Let's err on the side of noisiness for now.
"""
import re
import numpy as np
from collections import OrderedDict
from typing import Set, List, Union

import pytest

import gym.spaces as spaces

from .constant import Param
import playtest.action as acn
from .test_state import MockState


class MockDecision(acn.BaseDecision):
    """Mock decision for testing rendering
    """

    DECIDE_BOOLEAN = "bool"
    DECIDE_INT_IN_RANGE = "range"
    DECIDE_INT_IN_SET = "set"

    all_decision = OrderedDict(
        [
            # Unit: Card ID
            (DECIDE_BOOLEAN, acn.ActionBool()),
            (DECIDE_INT_IN_RANGE, acn.ActionIntInRange(0, 10)),
            (DECIDE_INT_IN_SET, acn.ActionIntInSet({1, 3, 5})),
        ]
    )


@pytest.fixture
def md() -> MockDecision:
    md = MockDecision(
        {
            MockDecision.DECIDE_BOOLEAN: True,
            MockDecision.DECIDE_INT_IN_SET: {3, 5},
            MockDecision.DECIDE_INT_IN_RANGE: (1, 3),
        },
        current_player_id=0,
    )
    return md


def test_decision_check_bool(md: MockDecision):
    """Test that we created out decisions correctly
    """

    # Converting decision
    bool_action = Action.from_str("bool()")
    assert md.is_valid(bool_action)


@pytest.xfail
def test_set_action(md: MockDecision):
    int_in_set_action = Action.from_str("set(2)")
    assert md.is_valid(int_in_set_action)

    invalid_action = Action.from_str("set(10)")
    assert not md.is_valid(invalid_action)


@pytest.xfail
def test_range_action(md: MockDecision):
    int_in_range_action = Action.from_str("range(2)")

    assert md.is_valid(int_in_range_action)

    invalid_action = Action.from_str("range(10)")
    assert not md.is_valid(invalid_action)


@pytest.xfail
def test_action_factor_action_space(md: MockDecision):
    assert md.number_of_actions == 3


@pytest.xfail
def test_md_to_action_ing(md: MockDecision):
    """Action only takes int, so we have to convert this
    into a valid int somehow!
    """
    action_map = md.get_action_map()
    assert action_map == [
        (MockDecision.DECIDE_BOOLEAN, 0, 1),
        (MockDecision.DECIDE_INT_RANGE, 1, 7),
        (MockDecision.DECIDE_INT_IN_SET, 7, 10),
    ]

    action_value = md.to_int({MockDecision.DECIDE_INT_IN_SET: 2})
    assert isinstance(action_value, int), "Convert action to np.int"
    assert action_value == 8
    action = md.from_int(action_value)
    assert action == Action.from_str("bool()")
    action = md.from_int(5)
    assert action == Action.from_str("set(2)")
    # since wait is the first item in MockActionFactory
    action = md.from_int(0)
    assert action == Action.from_str("range(10)")
