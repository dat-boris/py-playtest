from typing import Dict, Callable, Tuple, Optional
import enum
import abc
from collections import defaultdict

from .state import FullState
from .action import BaseDecision, ActionInstance

# Gamestate must be of enum
GameState = enum.Enum

# State, ActionDecision, GameState, next player
TypeHandlerReturn = Tuple[FullState, Optional[BaseDecision], GameState, Optional[int]]


class GameHandler:
    """A class which contains a mapping of various handler function
    """

    @abc.abstractproperty
    def handler(self) -> Dict[enum.Enum, Callable]:
        raise NotImplementedError()

    def get_handler(
        self, game_state: enum.Enum
    ) -> Callable[[FullState, Optional[ActionInstance]], TypeHandlerReturn]:
        return self.handler[game_state]


class __RewardSingleton:
    """A singleton class for logging any rewards"""

    points_stored: defaultdict

    def __init__(self):
        self.points_stored = defaultdict(int)

    def reset(self):
        self.points_stored = defaultdict(int)

    def add_reward(self, player_id: int, reward: int):
        self.points_stored[player_id] += reward

    def pop_reward_for_player(self, player_id: int) -> int:
        points = self.points_stored[player_id]
        self.points_stored[player_id] = 0
        return points


RewardLogger = __RewardSingleton()
