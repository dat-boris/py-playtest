import pytest

import numpy as np
import gym.spaces as spaces

from playtest.env import GameWrapperEnvironment
from playtest.action import InvalidActionError, ActionWait, ActionWaitRange


from .utils import Reward
from .env import BlackjackEnvironment
from .action import (
    ActionBetRange,
    ActionHit,
    ActionBet,
    ActionHitRange,
    ActionSkipRange,
)


AGENT_COUNT = 2


@pytest.fixture
def env() -> GameWrapperEnvironment:
    env = BlackjackEnvironment()
    return env


def test_reset(env):
    array = env.reset()
    assert isinstance(array, list)
    assert isinstance(array[0], np.ndarray)


def test_obs_space(env):
    space = env.observation_space
    assert isinstance(space, spaces.Tuple)
    assert len(space) == 2, "Observe space contains both action and observation"
    assert space[0]["wait"], "Action space contains wait"

    assert spaces.flatdim(space[1]) == (
        52
        + 52  # discarded
        + 2  # player hand
        + (1 + 1) * (AGENT_COUNT - 1)  # player bank + bet  # other player hands
    ), "Observation space is of right shape"


def test_action_space(env):
    space = env.action_space
    assert space
    assert spaces.flatdim(space) == 4


def test_reward(env: GameWrapperEnvironment):
    assert env.reward_range[0] < 0
    assert env.reward_range[1] > 0


def test_step_needs_action(env: GameWrapperEnvironment):
    env.reset()
    corrupt_input = np.array([-99, -99, -99, -99])
    with pytest.raises(InvalidActionError):
        _, _, _, _ = env.step([corrupt_input, corrupt_input])


def test_invalid_action(env: GameWrapperEnvironment):
    """Ensure that invalid action will get punished

    And also observation should represent the accepted action
    """
    env.reset()
    assert env.next_accepted_action == [ActionBetRange(0, 10)]
    assert env.next_player == 0

    hit_numpy_value = env.action_factory.to_numpy(ActionHit())
    bet3_numpy_value = env.action_factory.to_numpy(ActionBet("3"))
    wait_numpy_value = env.action_factory.to_numpy(ActionWait())

    obs, reward, _, _ = env.step([hit_numpy_value, bet3_numpy_value])

    assert reward[0] < 0
    assert reward[1] < 0
    _, reward, _, _ = env.step([wait_numpy_value, wait_numpy_value])
    assert reward[0] < 0
    assert reward[1] == Reward.VALID_ACTION
    _, reward, _, _ = env.step([bet3_numpy_value, bet3_numpy_value])
    assert reward[0] == Reward.BETTED
    assert reward[1] < 0


def test_continuous_invalid_action(env: GameWrapperEnvironment):
    """Ensure that continuous invalid action will lead to termination"""
    env.reset()
    assert env.next_accepted_action == [ActionBetRange(0, 10)]
    assert env.next_player == 0
    termination = [False, False]
    hit_numpy_value = env.action_factory.to_numpy(ActionHit())
    bet3_numpy_value = env.action_factory.to_numpy(ActionBet("3"))

    for _ in range(env.max_continuous_invalid_inputs):
        obs, reward, termination, _ = env.step([hit_numpy_value, bet3_numpy_value])
    assert termination[0]
    assert reward[0] == Reward.INVALID_ACTION


def test_step(env: GameWrapperEnvironment):
    env.reset()

    action_factory = env.action_factory

    # Note round 1: only one agent we care about!
    assert env.next_player == 0
    assert env.next_accepted_action == [ActionBetRange(0, 10)]

    hit_numpy_value = env.action_factory.to_numpy(ActionHit())
    bet1_numpy_value = env.action_factory.to_numpy(ActionBet("1"))
    wait_numpy_value = env.action_factory.to_numpy(ActionWait())

    obs, reward, terminal, info = env.step([bet1_numpy_value, wait_numpy_value])
    assert len(obs) == AGENT_COUNT
    assert len(reward) == AGENT_COUNT
    assert len(terminal) == AGENT_COUNT
    assert all([r >= 0 for r in reward]), f"not contain negative {reward}"

    # given the observation, we should be able to flatten it
    # and obtain reasonable result
    obs_space = env.observation_space
    flatten_data = obs[0]
    assert flatten_data.size == spaces.flatdim(obs_space)

    # Now we need to action again
    assert env.next_player == 0
    assert env.next_accepted_action == [ActionHitRange(), ActionSkipRange()]
    obs, reward, terminal, info = env.step([hit_numpy_value, wait_numpy_value,])
    assert len(obs) == AGENT_COUNT
    assert len(reward) == AGENT_COUNT
    assert reward[0] == Reward.HITTED
    assert len(terminal) == AGENT_COUNT
    assert all([r >= 0 for r in reward]), f"contain negative {reward}"
