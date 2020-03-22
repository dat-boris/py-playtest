from typing import (
    List,
    Tuple,
    Generator,
    Sequence,
    Type,
    Dict,
    Optional,
    Generic,
    TypeVar,
)
import re
import warnings
import numpy as np

import gym.spaces as spaces

from .constant import Param, Reward
from .state import FullState
from .action import ActionFactory, ActionInstance, ActionRange
from .logger import Announcer


class Player:
    id: int

    def __init__(self, id):
        self.id = id


S = TypeVar("S", bound=FullState)
AF = TypeVar("AF", bound=ActionFactory)
P = TypeVar("P", bound=Param)


class Game(Generic[S, AF, P]):

    base_state: Type[S]

    announcer: Announcer
    param: P
    state: S
    action_factory: AF
    players: List[Player]
    last_player_reward: int

    def __init__(self, param: P):
        self.param = param
        self.announcer = Announcer()
        self.last_player_reward = Reward.DEFAULT
        if getattr(self, "players", None) is None:
            assert param.number_of_players > 0, "Must have some players!"
            self.players = [Player(i) for i in range(param.number_of_players)]
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

    def reset(self) -> S:
        """Reset the game

        This reset the game to the first initial observation of the first
        player move.  Note that this might mean that this include some
        setup (e.g. dealing initial cards to player).

        The return of State is intentional, and allow the environment to return
        the initial observation.
        """
        self.a.say("Resetting game.")
        self.s.reset()
        return self.s

    def start(
        self,
    ) -> Generator[
        # return: player_id, possible action, last_player_reward
        Tuple[int, Sequence[ActionRange], int],
        ActionInstance,  # receive: action
        None,  # terminal
    ]:
        raise NotImplementedError()

    def set_last_player_reward(self, reward):
        self.last_player_reward = reward

    def get_player_action(
        self,
        player_id: int,
        accepted_action: Optional[Sequence[ActionRange]] = None,
        accepted_range: Optional[Sequence[Type[ActionRange]]] = None,
    ) -> Generator[
        # return: player_id, possible action, last_player_reward
        Tuple[int, Sequence[ActionRange], int],
        ActionInstance,  # receive: action
        ActionInstance,  # terminal
    ]:
        """Getting player action

        Note that we must use yield from to call this function
        """
        if accepted_action is None:
            accepted_action = self.action_factory.get_actionable_actions(
                self.state, player_id, accepted_range=accepted_range
            )
        else:
            warnings.warn("accepted_action will be removed", DeprecationWarning)
        action = yield (player_id, accepted_action, self.last_player_reward)
        return action

    ###########
    # Shorthands
    ###########
    @property
    def a(self):
        return self.announcer

    @property
    def s(self):
        return self.state
