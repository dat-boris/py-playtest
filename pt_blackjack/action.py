import numpy as np
import enum
from collections import OrderedDict

import gym.spaces as spaces

import playtest.action as acn
from playtest.action_range import action_bool, action_int

from pt_blackjack.constant import Param
from pt_blackjack.state import State


class ActionName(enum.Enum):
    # TODO: this does not gets mapped
    # WAIT = "wait"
    SKIP = "skip"
    HIT = "hit"
    BET = "bet"


class ActionDecision(acn.BaseDecision):
    action_enum = ActionName
    decision_ranges = OrderedDict(
        [
            (ActionName.SKIP, action_bool.ActionBooleanRange(ActionName.SKIP, None),),
            (ActionName.HIT, action_bool.ActionBooleanRange(ActionName.HIT, None),),
            # TODO: assert max bank for now as 100
            (ActionName.BET, action_int.ActionIntInRange(ActionName.BET, (0, 100)),),
        ]
    )

