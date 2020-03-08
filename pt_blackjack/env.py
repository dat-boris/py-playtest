from playtest.env import GameWrapperEnvironment

from .constant import Param
from .game import Blackjack

AGENT_COUNT = 2

__game = Blackjack(Param(number_of_players=AGENT_COUNT))
# TODO: make this environment a meta-class
BlackjackEnvironment = lambda **kwargs: GameWrapperEnvironment(__game)
