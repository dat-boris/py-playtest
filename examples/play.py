#!/usr/bin/env python
"""Run interactive agent for argparse
"""
import argparse
from pprint import pprint

import gym

from playtest.env import GameWrapperEnvironment, HumanAgent, EnvironmentInteration
from pt_blackjack import Blackjack, Param

AGENT_COUNT = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Agent for ma-gym")
    parser.add_argument(
        "--episodes", type=int, default=1, help="episodes (default: %(default)s)"
    )
    args = parser.parse_args()

    __game = Blackjack(Param(number_of_players=AGENT_COUNT))
    env: GameWrapperEnvironment = GameWrapperEnvironment(__game)

    agents = [HumanAgent(env) for i in range(env.n_agents)]

    game = EnvironmentInteration(env, agents, episodes=args.episodes)
    game.play()
