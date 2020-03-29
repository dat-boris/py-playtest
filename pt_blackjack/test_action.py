import re
import numpy as np
import pytest

import gym.spaces as spaces

from playtest.action import InvalidActionError

from pt_blackjack.action import ActionBetRange, ActionBet
from pt_blackjack.test_state import state


@pytest.fixture
def action_range(state):
    return ActionBetRange(state, player_id=0)


def test_action_range_valid(action_range):
    # Note that upper bound is inclusive
    action = ActionBet(3)
    assert action_range.is_valid(action)


def test_action_range_invalid(action_range):
    """Test that action has an open upper bound"""
    action = ActionBet(10)
    assert not action_range.is_valid(action)


def test_action_str(action_range):
    action = ActionBet(3)
    assert str(action_range) == "bet(1->10)"
    assert str(action) == "bet(3)"
    assert repr(action_range) == "bet(1->10)"
    assert repr(action) == "bet(3)"
    action_reversed = ActionBet.from_str("bet(3)")
    assert action_reversed == action


def test_action_range_numpy(action_range: ActionBetRange):
    assert action_range.get_number_of_distinct_value() == 19
    assert action_range.to_numpy_data().tolist() == [1, 10]


def test_action_numpy():
    action = ActionBet(3)
    assert action.get_number_of_distinct_value() == 19
    assert action.to_int() == 3 - 2
    with pytest.raises(InvalidActionError):
        ActionBet(1)
