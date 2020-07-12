import os
import pytest

from playtest.agents import KerasDQNAgent, train_agents
from playtest.env import GameWrapperEnvironment, EnvironmentInteration

from .test_env import env_allow_invalid
from .test_human import get_patched_agent


AGENT_FILENAME = "example_agent.h5f"


def test_training(env_allow_invalid):
    env = env_allow_invalid
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


def test_playing(env_allow_invalid):
    env = env_allow_invalid

    assert env.n_agents == 2
    assert os.path.exists(AGENT_FILENAME), "No agent file - training test first"
    new_agent = KerasDQNAgent(env, weight_file=AGENT_FILENAME)

    # This creates a mock human agent
    agents = [
        get_patched_agent(env, ["bet(3)", "hit()", "skip()", "bet(3)"]),
        new_agent,
    ]

    # Let's play 4 rounds of game!
    game = EnvironmentInteration(env, agents, rounds=4)

    state = game.env.state
    assert len(state.discarded) == 0
    assert state.players[0].bank.value == [10]

    game.play()

    assert (
        len(state.players[0].hand) == 3 or len(state.discarded) == 4
    ), "Game progressed"
    assert True, "Game exists"
