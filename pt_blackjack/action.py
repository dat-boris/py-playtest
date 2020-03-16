import numpy as np

import gym.spaces as spaces

from pt_blackjack.state import State
from playtest.action import (
    ActionFactory as BaseActionFactory,
    ActionSingleValue,
    ActionSingleValueRange,
    ActionWaitRange,
    ActionBoolean,
    ActionBooleanRange,
)


class ActionHit(ActionBoolean):
    key = "hit"


class ActionHitRange(ActionBooleanRange[ActionHit]):
    instance_class = ActionHit


class ActionSkip(ActionBoolean):
    key = "skip"


class ActionSkipRange(ActionBooleanRange[ActionSkip]):
    instance_class = ActionSkip


# Sentinel Action instance for comparison
ACTION_SKIP = ActionSkip()
ACTION_HIT = ActionHit()


class ActionBet(ActionSingleValue):
    key = "bet"

    minimum_value = 0
    maximum_value = 0xFF


class ActionBetRange(ActionSingleValueRange[ActionBet]):
    instance_class = ActionBet

    def __init__(self, state: State, player_id: int):
        ps = state.get_player_state(player_id)
        self.lower = 1
        self.upper = ps.bank.amount


class ActionFactory(BaseActionFactory):
    range_classes = [
        ActionWaitRange,
        ActionHitRange,
        ActionSkipRange,
        ActionBetRange,
    ]
