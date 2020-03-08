import re
import numpy as np
import pytest

import gym.spaces as spaces


from .action import ActionBetRange


@pytest.fixture
def action_range():
    return ActionBetRange(1, 3)


def test_action_range_valid(action_range):
    # Note that upper bound is inclusive
    action = ActionBet(3)
    assert action_range.is_valid(action)


def test_action_range_invalid(action_range):
    """Test that action has an open upper bound"""
    action = ActionBet(4)
    assert not action_range.is_valid(action)


def test_action_str(action_range):
    action = ActionBet(3)
    assert str(action_range) == "bet(1->3)"
    assert str(action) == "bet(3)"
    assert repr(action_range) == "bet(1->3)"
    assert repr(action) == "bet(3)"
    action_reversed = ActionBet.from_str("bet(3)")
    assert action_reversed == action


def test_action_range_numpy(action_range: ActionBetRange):
    assert action_range.action_space_possible
    assert action_range.to_numpy_data().tolist() == [1, 3]


def test_action_numpy():
    action = ActionBet(3)
    assert action.action_space
    assert action.to_numpy_data() == [3]
