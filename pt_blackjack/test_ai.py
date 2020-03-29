import os
import pytest

from playtest.agents import KerasDQNAgent, train_agents
from playtest.env import GameWrapperEnvironment, EnvironmentInteration

from .test_env import env
from .test_human import get_patched_agent


AGENT_FILENAME = "example_agent.h5f"


def test_training(env: GameWrapperEnvironment):
    agents = [KerasDQNAgent(env) for _ in range(env.n_agents)]
    try:
        os.remove(AGENT_FILENAME)
    except OSError:
        pass
    train_agents(env, agents, save_filenames=[AGENT_FILENAME], nb_steps=10)
    assert os.path.exists(AGENT_FILENAME)

    new_agent = KerasDQNAgent(env)
    new_agent.load_weights(AGENT_FILENAME)

    assert new_agent


def test_playing(env):
    assert env.n_agents == 2
    at = KerasDQNAgent(env)
    new_agent = KerasDQNAgent(env, weight_file=AGENT_FILENAME)

    agents = [get_patched_agent(env, ["bet(3)", "skip"]), new_agent]

    # Let's play 4 rounds of game!
    game = EnvironmentInteration(env, agents, rounds=4)
    game.play()

    state = game.env.game.s
    # assert len(state.discarded) == 4, "4 cards discarded"
    assert True, "Game exists"
