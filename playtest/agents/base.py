from rl.core import Agent

from playtest.env import GameWrapperEnvironment


# TODO: probably should not inherit directly from rl-keras
class BaseAgent(Agent):
    """Simple wrapper around rl.keras agent

    That provides ability to save and load weights
    """

    env: GameWrapperEnvironment

    def __init__(self, env):
        self.env = env
