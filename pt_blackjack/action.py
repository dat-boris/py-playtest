import numpy as np

import gym.spaces as spaces

from playtest.action import (
    ActionInstance,
    ActionRange,
    ActionFactory as BaseActionFactory,
    ActionSingleValue,
    ActionSingleValueRange,
    ActionWaitRange,
)


class ActionHit(ActionInstance):
    key = "hit"


class ActionHitRange(ActionRange):
    instance_class = ActionHit


class ActionSkip(ActionInstance):
    key = "skip"


class ActionSkipRange(ActionRange):
    instance_class = ActionSkip


# Sentinel Action instance for comparison
ACTION_SKIP = ActionSkip()
ACTION_HIT = ActionHit()


class ActionBet(ActionSingleValue):
    key = "bet"

    # possible action space for the action
    action_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int8)


class ActionBetRange(ActionSingleValueRange):
    instance_class = ActionBet
    upper: int
    lower: int

    min_lower = 1
    max_upper = 100

    action_space_possible = spaces.Box(low=1, high=100, shape=(2,), dtype=np.int8)


class ActionFactory(BaseActionFactory):
    range_classes = [
        ActionWaitRange,
        ActionHitRange,
        ActionSkipRange,
        ActionBetRange,
    ]
