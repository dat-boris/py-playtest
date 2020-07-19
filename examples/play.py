#!/usr/bin/env python
"""Run interactive agent for argparse
"""
import os, sys
import argparse
from pprint import pprint

import gym

sys.path.insert(0, os.getcwd())

from playtest.env import GameWrapperEnvironment, EnvironmentInteration
from playtest.agents import HumanAgent, KerasDQNAgent

from pt_blackjack.state import State
import pt_blackjack.game as gm
import pt_blackjack.action as acn

AGENT_COUNT = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Agent for ma-gym")
    parser.add_argument(
        "--episodes", type=int, default=1, help="episodes (default: %(default)s)"
    )
    parser.add_argument(
        "--bot",
        type=str,
        default="blackjack_dnq.h5f",
        help="Input agent file (default: %(default)s)",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        default=False,
        help="If we want to play against ai",
    )
    args = parser.parse_args()

    __game = gm.BlackjackHandler(number_of_players=AGENT_COUNT)
    env: GameWrapperEnvironment = GameWrapperEnvironment(__game)

    second_agent = (
        KerasDQNAgent(env, weight_file=args.bot) if args.ai else HumanAgent(env)
    )

    agents = [HumanAgent(env), second_agent]

    game = EnvironmentInteration(env, agents, episodes=args.episodes)
    game.play()
