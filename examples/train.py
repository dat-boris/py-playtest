#!/usr/bin/env python
"""Run interactive agent for argparse
"""
import argparse
from pprint import pprint

import gym

from playtest.env import GameWrapperEnvironment
from playtest.agents import KerasDQNAgent, train_agents

from pt_blackjack import Blackjack, Param

AGENT_COUNT = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Agent for ma-gym")
    parser.add_argument(
        "--episodes", type=int, default=100000, help="episodes (default: %(default)s)"
    )
    parser.add_argument("--pdb", action="store_true", help="Add debugger on error")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print verbose message"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="blackjack_dnq.h5f",
        help="Output agent file (default: %(default)s)",
    )
    args = parser.parse_args()

    __game = Blackjack(Param(number_of_players=AGENT_COUNT), verbose=args.verbose)
    env: GameWrapperEnvironment = GameWrapperEnvironment(__game, verbose=args.verbose)

    agents = [KerasDQNAgent(env) for _ in range(env.n_agents)]

    train_agents(
        env, agents, save_filenames=[args.output], nb_steps=args.episodes, is_pdb=True,
    )
