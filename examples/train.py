#!/usr/bin/env python
"""Run interactive agent for argparse
"""
import os, sys
import argparse
from pprint import pprint

import gym

sys.path.insert(0, os.getcwd())

from playtest.env import GameWrapperEnvironment
from playtest.agents import KerasDQNAgent, train_agents

from .constant import Reward, Param
from pt_blackjack.state import State
import pt_blackjack.game as gm
import pt_blackjack.action as acn

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

    env: GameWrapperEnvironment = GameWrapperEnvironment(
        gm.BlackjackHandler(),
        State(Param(number_of_players=AGENT_COUNT)),
        gm.GameState.start,
        acn.ActionDecision,
        verbose=args.verbose,
    )

    agents = [KerasDQNAgent(env) for _ in range(env.n_agents)]

    train_agents(
        env, agents, save_filenames=[args.output], nb_steps=args.episodes, is_pdb=True,
    )
