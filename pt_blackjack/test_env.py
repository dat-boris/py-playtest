import pytest

import numpy as np
import gym.spaces as spaces

from playtest.env import GameWrapperEnvironment
from playtest.action import InvalidActionError, ActionInstance


from .constant import Reward, Param
from pt_blackjack.state import State
import pt_blackjack.game as gm
import pt_blackjack.action as acn

AGENT_COUNT = 2


@pytest.fixture
def env() -> GameWrapperEnvironment:
    # TODO: setting up the handler
    env = GameWrapperEnvironment(
        gm.BlackjackHandler(),
        State(Param(number_of_players=AGENT_COUNT)),
        gm.GameState.start,
        acn.ActionDecision,
        allow_invalid=False,
    )
    return env


@pytest.fixture
def env_allow_invalid() -> GameWrapperEnvironment:
    # TODO: setting up the handler
    env = GameWrapperEnvironment(
        gm.BlackjackHandler(),
        State(Param(number_of_players=AGENT_COUNT)),
        gm.GameState.start,
        acn.ActionDecision,
        allow_invalid=True,
    )
    return env


def test_reset(env):
    array = env.reset()
    assert isinstance(array, list)
    assert isinstance(array[0], np.ndarray)

    # Now check the next_action_array is expected
    decision: acn.ActionDecision = env.next_accepted_action
    assert isinstance(decision, acn.ActionDecision)
    bet_action = ActionInstance(acn.ActionName.BET, 5)
    assert decision.is_legal(bet_action)


def test_obs_space(env):
    array = env.reset()
    space = env.observation_space
    assert isinstance(space, spaces.Tuple)
    assert len(space) == 2, "Observe space contains both action and observation"
    assert space[0][acn.ActionName.HIT.name], "Action space contains hit"
    assert space[1]["self"]["hand"], "You can see your own hand"
    assert "hand" not in space[1]["others"][0], "You cannot see other hands"

    assert spaces.flatdim(space[1]) >= (
        52 * 2
        + 52 * 2  # discarded
        + 2  # player hand
        # player bank + bet  # other player hands
        + (1 + 1) * (AGENT_COUNT - 1)
    ), "Observation space is of right shape"


def test_action_space(env):
    array = env.reset()
    space = env.action_space
    assert space
    # This is a large space, as bet can have a large linear space
    assert spaces.flatdim(space) >= 23


# @pytest.mark.xfail
# def test_reward(env: GameWrapperEnvironment):
#     # TODO: reward was not implemented
#     assert env.reward_range[0] < 0
#     assert env.reward_range[1] > 0


def __action_int(env: GameWrapperEnvironment, action: ActionInstance) -> int:
    assert env.next_accepted_action is not None
    action_int = env.next_accepted_action.to_int(action)
    return action_int


def test_step_invalid_action(env_allow_invalid):
    env = env_allow_invalid
    env.reset()
    corrupt_input = -99
    # Given completely invalid input, give the error
    with pytest.raises(KeyError):
        _, _, _, _ = env.step([corrupt_input, corrupt_input])

    # Given a illegal input, give the error
    assert len(env.continuous_invalid_inputs) == 0
    illegal_action_int = __action_int(env, ActionInstance(acn.ActionName.HIT, True))
    _, _, _, _ = env.step([illegal_action_int, None])
    # Ensure that we have set the invalid action
    assert len(env.continuous_invalid_inputs) == 1
    assert env.next_player == 0

    # Now let's keep giving the player bad action
    # We already have one invalid action sent, so -1 count
    for _ in range(env.max_continuous_invalid_inputs - 1):
        obs, reward, _, _ = env.step([illegal_action_int, None])
        # TODO: not yet punished
        # assert reward[0] < 0, "Player is punished"

    state = env.state
    bet_made = state.get_player_state(0).bet.value[0]
    assert bet_made >= 0
    assert len(env.continuous_invalid_inputs) == 0


def test_step(env: GameWrapperEnvironment):
    env.reset()

    # Note round 1: only one agent we care about!
    assert env.next_player == 0

    bet_action_int = __action_int(env, ActionInstance(acn.ActionName.BET, 3))

    # Player 1 bet
    obs, reward, terminal, info = env.step([bet_action_int, None])
    assert len(obs) == AGENT_COUNT
    assert len(reward) == AGENT_COUNT
    assert len(terminal) == AGENT_COUNT
    # TODO: reward not set
    # assert all([r >= 0 for r in reward]), f"not contain negative {reward}"

    # given the observation, we should be able to flatten it
    # and obtain reasonable result
    obs_space = env.observation_space
    flatten_data = obs[0]
    assert flatten_data.size == spaces.flatdim(obs_space)  # type: ignore

    # Now we need to action again
    assert env.next_player == 0
    hit_action_int = __action_int(env, ActionInstance(acn.ActionName.HIT, True))

    obs, reward, terminal, info = env.step([hit_action_int, None])
    assert len(obs) == AGENT_COUNT
    assert len(reward) == AGENT_COUNT
    assert len(terminal) == AGENT_COUNT
    # TODO: reward not set
    # assert reward[0] == Reward.HITTED
    # assert all([r >= 0 for r in reward]), f"contain negative {reward}"
