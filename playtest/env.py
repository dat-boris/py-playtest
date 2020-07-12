import logging
from typing import Tuple, Generator, Optional, List, Dict, Sequence, Type, Any, Callable
from pprint import pprint
import warnings
import enum

import gym.spaces as spaces

# TODO: Ignore old version of tensorflow warning
warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa
warnings.simplefilter(action="ignore", category=DeprecationWarning)  # noqa

import numpy as np
import gym
import gym.utils.seeding as seeding
import gym.spaces as spaces
from rl.core import Agent

from .game import GameHandler, TypeHandlerReturn
from .state import FullState
from .constant import Reward
from .action import BaseDecision, ActionInstance


class TooManyInvalidActions(Exception):
    """Raise when the player ran too many exception"""

    pass


class GameWrapperEnvironment(gym.Env):
    """A wrapper which converts playtest into an environment

    This is built based on reference of the cartpole elements
    gym: envs/classic_control/cartpole.py
    """

    metadata = {"render.modes": ["human"]}

    game_handler: GameHandler
    state: FullState
    decision_class: Type[BaseDecision]

    current_state: enum.Enum
    next_player: int
    next_accepted_action: Optional[BaseDecision]

    # Number of times input is invalid
    continuous_invalid_inputs: List[ActionInstance]

    # Number of times before we choose a random action and continue
    max_continuous_invalid_inputs: int = 5

    verbose: bool
    # If not allow invalid, raise exception when action is invalid
    allow_invalid: bool

    def __init__(
        self,
        gh: GameHandler,
        s: FullState,
        start_state: enum.Enum,
        decision_class: Type[BaseDecision],
        verbose=True,
        allow_invalid=True,
    ):
        # Categories of information required
        self.state = s
        self.game_handler = gh
        self.start_state = start_state
        self.decision_class = decision_class
        self.verbose = verbose
        self.allow_invalid = allow_invalid

        # Now setting internal state flags
        self.next_player = 0
        self.current_state = start_state
        self.next_accepted_action = None

        # Initialize other status
        self.continuous_invalid_inputs = []

    @property
    def n_agents(self) -> int:
        return self.state.number_of_players

    @property
    def action_space(self) -> spaces.Space:
        """Return the size of action space
        """
        return spaces.MultiBinary(self.decision_class.get_number_of_actions())

    def __get_all_players_observation_with_action(
        self, state: FullState, decision: BaseDecision
    ) -> List[np.ndarray]:
        """This return a map of the action space,
        with a multi-discrete action space for checking next_accepted_action
        """
        obs = [None] * self.n_agents

        next_player: int = self.next_player
        action_obs: Dict[str, np.ndarray] = decision.action_range_to_numpy()
        player_obs_space = self.state.to_player_data(next_player, for_numpy=True)

        # TODO: only set action for the next player
        obs[next_player] = spaces.flatten(
            # Note this includes action observation
            self.observation_space,
            [action_obs, player_obs_space],
        )
        return obs

    @property
    def observation_space(self) -> spaces.Space:
        """Get a combination of action and observation space from the game
        """
        obs_space = spaces.Tuple(
            [
                self.decision_class.action_space_possible(),
                self.state.get_observation_space_from_player(),
            ]
        )
        return obs_space

    @property
    def reward_range(self) -> Tuple[int, int]:
        low, high = self.game.reward_range
        assert low >= Reward.INVALID_ACTION, "Punish reward must be higher than game"
        return low, high

    def reset(self) -> List[np.ndarray]:
        """Return instance of environment
        """
        self.state.reset()

        handler_func: Callable[
            [FullState, Optional[ActionInstance]], TypeHandlerReturn
        ] = self.game_handler.get_handler(self.current_state)
        new_state, decision, next_game_state, next_player = handler_func(
            self.state, None
        )

        self.state = new_state
        self.next_accepted_action: Optional[BaseDecision] = decision
        assert (
            self.next_accepted_action is not None
        ), "Must provide decision on initialization"
        self.current_state = next_game_state
        assert next_player is not None
        self.next_player = next_player

        self.continuous_invalid_inputs = []

        return self.__get_all_players_observation_with_action(
            self.state, self.next_accepted_action
        )

    def __invalid_action_return(
        self, action: ActionInstance
    ) -> Tuple[
        List[Optional[spaces.Space]],  # observations
        List[int],  # rewards
        List[bool],  # terminals
        Dict,  # info
    ]:
        if not self.allow_invalid:
            raise RuntimeError(
                f"Invalid action seen {action}.  Current state: {self.current_state}"
            )
        self.continuous_invalid_inputs.append(action)
        if len(self.continuous_invalid_inputs) >= self.max_continuous_invalid_inputs:
            err_msg = (
                f"Getting continue bad input: {self.continuous_invalid_inputs}."
                "Going to pick a random action"
            )
            if self.verbose:
                logging.warn(err_msg)
            self.continuous_invalid_inputs = []
            raise TooManyInvalidActions(err_msg)
        logging.warning(f"🙅‍♂️ Action {action} is not valid.")
        assert self.next_accepted_action is not None
        return (
            self.__get_all_players_observation_with_action(
                self.state, self.next_accepted_action
            ),
            # TODO: setup rewards
            [0] * self.n_agents,
            [False] * self.n_agents,
            {},
        )

    def step(
        self, agents_action: List[Optional[int]]
    ) -> Tuple[
        List[Optional[spaces.Space]],  # observations
        List[int],  # rewards
        List[bool],  # terminals
        Dict,  # info
    ]:
        """Given the action, forward to the player
        """
        # TODO: not setting the rewards
        rewards = [0] * self.n_agents

        # Given a list of action, map action into necessary int
        current_player_action_int = agents_action[self.next_player]
        assert current_player_action_int is not None
        assert self.next_accepted_action is not None
        action_to_send: ActionInstance = self.next_accepted_action.from_int(
            current_player_action_int
        )

        # Check if action is legal
        if not self.next_accepted_action.is_legal(action_to_send):
            try:
                return self.__invalid_action_return(action_to_send)
            except TooManyInvalidActions:
                action_to_send = self.next_accepted_action.pick_random_action()

        # Now sending the necessary actions
        handler_func: Callable[
            [FullState, Optional[ActionInstance]], TypeHandlerReturn
        ] = self.game_handler.get_handler(self.current_state)
        new_state, decision, next_game_state, next_player = handler_func(
            self.state, action_to_send
        )

        self.state = new_state
        self.next_accepted_action: Optional[BaseDecision] = decision
        assert (
            self.next_accepted_action is not None
        ), "Must provide decision on initialization"
        self.current_state = next_game_state
        assert next_player is not None
        self.next_player = next_player

        return (
            self.__get_all_players_observation_with_action(
                self.state, self.next_accepted_action
            ),
            # TODO: setup rewards
            [0] * self.n_agents,
            [False] * self.n_agents,
            {},
        )

    def render(self, mode="human"):
        """Render the relevant cards
        """
        pass

    def to_player_data(self, player_id: int) -> Dict:
        return self.state.to_player_data(player_id)

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        return


class EnvironmentInteration:
    """Represent a game that can be played"""

    env: GameWrapperEnvironment
    agents: List[Agent]
    episodes: int
    rounds: Optional[int]
    max_same_player: int

    def __init__(self, env, agents, episodes=1, rounds=None, max_same_player=20):
        assert len(agents) == env.n_agents
        self.env = env
        self.agents = agents
        self.episodes = episodes
        self.rounds = rounds
        self.max_same_player = max_same_player

    def play(self):
        env = self.env

        rounds = 0
        for ep_i in range(self.episodes):
            done_n = [False for _ in range(env.n_agents)]
            ep_reward = 0

            obs_n = env.reset()
            env.render()

            same_player_count = 0
            last_player = None
            while not all(done_n):
                player_id = env.next_player
                assert player_id is not None
                if last_player == player_id:
                    same_player_count += 1
                    if same_player_count > self.max_same_player:
                        raise RuntimeError(
                            f"Same player {player_id} had been "
                            f"playing more than {self.max_same_player} rounds"
                        )

                action_n = [None] * env.n_agents
                # print(f"Player {player_id} taking action...")

                action_taken = self.agents[player_id].forward(obs_n[player_id])
                # print(f"Action taken: {action_taken}")
                assert isinstance(
                    int(action_taken), int
                ), f"Forward agent {self.agents[player_id]} should return an integer. (Got: {action_taken.__class__})"

                action_n[player_id] = action_taken

                obs_n, reward_n, done_n, _ = env.step(action_n)
                ep_reward += sum(reward_n)
                env.render()

                last_player = player_id

                # Check how many rounds we have done.
                rounds += 1
                if self.rounds is not None and rounds >= self.rounds:
                    print(f"Reached {rounds} rounds. exiting.")
                    env.close()
                    return

            print("Episode #{} Reward: {}".format(ep_i, ep_reward))
        env.close()
