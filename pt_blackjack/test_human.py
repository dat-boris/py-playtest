from playtest.agents import HumanAgent
from playtest.env import EnvironmentInteration

# import fixture
from .test_env import env


def get_patched_agent(env, mock_inputs):
    h = HumanAgent(env)
    h.get_input = lambda prompt: mock_inputs.pop(0)
    return h


def test_human(env):
    """Testing if human interaction is working
    """
    assert env.n_agents == 2
    agents = [get_patched_agent(env, ["bet(3)", "skip"]) for i in range(env.n_agents)]

    # Let's play 4 rounds of game!
    game = EnvironmentInteration(env, agents, rounds=4)
    game.play()

    state = game.env.game.s
    assert len(state.discarded) == 4, "4 cards discarded"
