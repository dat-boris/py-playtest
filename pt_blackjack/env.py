from playtest.envs.blackjack.game import Blackjack, Param
from playtest.core.env import GameWrapperEnvironment

AGENT_COUNT = 2

__game = Blackjack(Param(number_of_players=AGENT_COUNT))
# TODO: make this environment a meta-class
BlackjackEnvironment = lambda **kwargs: GameWrapperEnvironment(__game)
