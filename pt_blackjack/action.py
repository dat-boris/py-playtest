import numpy as np

import gym.spaces as spaces

from pt_blackjack.constant import Param
from pt_blackjack.state import State
from playtest.action import (
    ActionFactory as BaseActionFactory,
    ActionSingleValue,
    ActionSingleValueRange,
    ActionWaitRange,
    ActionBoolean,
    ActionBooleanRange,
)


class ActionHit(ActionBoolean[State]):
    key = "hit"

    def resolve(self, s, player_id, a=None):
        # TODO: move action resolve in here
        pass


class ActionHitRange(ActionBooleanRange[ActionHit, State]):
    instance_class = ActionHit

    def __init__(self, state: State, player_id: int):
        super().__init__(state, player_id)


class ActionSkip(ActionBoolean[State]):
    key = "skip"

    def resolve(self, s, player_id, a=None):
        # TODO: move action resolve in here
        pass


class ActionSkipRange(ActionBooleanRange[ActionSkip, State]):
    instance_class = ActionSkip

    def __init__(self, state: State, player_id: int):
        super().__init__(state, player_id)


# Sentinel Action instance for comparison
ACTION_SKIP = ActionSkip()
ACTION_HIT = ActionHit()


class ActionBet(ActionSingleValue[State]):
    key = "bet"

    # Minimum bet
    minimum_value = 2
    maximum_value = Param.max_bank

    def resolve(self, s, player_id, a=None):
        # TODO: move action resolve in here
        pass


class ActionBetRange(ActionSingleValueRange[ActionBet, State]):
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
