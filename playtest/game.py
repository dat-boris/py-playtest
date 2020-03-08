from typing import List, Tuple, Generator, Sequence, Type, Dict, Optional
import re
import numpy as np

import gym.spaces as spaces

from .utils import Param, Reward
from .state import FullState
from .action import ActionFactory, ActionInstance, ActionRange
from .logger import Announcer


class Player:
    id: int

    def __init__(self, id):
        self.id = id


class Game:
    announcer: Announcer
    param: Param
    state: FullState
    action_factory: ActionFactory
    players: List[Player]

    def __init__(self, param):
        self.param = param
        self.announcer = Announcer()
        assert self.state, "Must set state before init"
        assert self.action_factory, "Must set action before init"

    def get_announcer(self) -> Announcer:
        return self.announcer

    @property
    def number_of_players(self) -> int:
        """Return number of players
        """
        return self.param.number_of_players

    @property
    def reward_range(self) -> Tuple[int, int]:
        return (Reward.INVALID_ACTION, Reward.WINNER)

    def reset(self):
        self.state.reset()

    def start(
        self,
    ) -> Generator[
        # return: player_id, possible action, last_player_reward
        Tuple[int, Sequence[ActionRange], int],
        ActionInstance,  # receive: action
        None,  # terminal
    ]:
        raise NotImplementedError()

    ###########
    # Shorthands
    ###########
    @property
    def a(self):
        return self.announcer

    @property
    def s(self):
        return self.state
