from typing import Dict, Callable, Tuple, Optional
import enum
import abc

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

