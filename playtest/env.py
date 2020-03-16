import logging
from typing import Tuple, Generator, Optional, List, Dict, Sequence, Type
from pprint import pprint
import warnings

# TODO: Ignore old version of tensorflow warning
warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa
warnings.simplefilter(action="ignore", category=DeprecationWarning)  # noqa

import numpy as np
import gym
import gym.spaces as spaces
from rl.core import Agent

from .game import Game
from .state import FullState
from .action import ActionFactory, ActionRange, ActionInstance, InvalidActionError
from .constant import Reward


class GameWrapperEnvironment(gym.Env):
    """A wrapper which converts playtest into an environment

    This is built based on reference of the cartpole elements
    gym: envs/classic_control/cartpole.py
    """

    metadata = {"render.modes": ["human"]}

    game: Game
    action_factory: ActionFactory
    state: FullState
    game_gen: Optional[Generator]
    next_player: int
    next_accepted_action: Optional[Sequence[ActionRange]]
    cached_space: Optional[spaces.Space]
    # Number of times input is invalid
    continuous_invalid_inputs: List[ActionInstance]
    max_continuous_invalid_inputs: int = 100

    def __init__(self, game: Game):
        self.game = game
        self.action_factory = game.action_factory
        self.state = game.state
        self.game_gen = None
        # NOTE: default to first player, avoid mistakes in None check
        self.next_player = 0
        self.next_accepted_action = None
        self.cached_space = None
        self.continuous_invalid_inputs = []

    @property
    def n_agents(self) -> int:
        return self.game.number_of_players

    @property
    def action_space(self) -> spaces.Space:
        """Return the size of action space
        """
        return self.action_factory.action_space

    def __get_all_players_observation_with_action(self) -> List[np.ndarray]:
        """This return a map of the action space,
        with a multi-discrete action space for checking next_accepted_action
        """
        obs = [None] * self.game.number_of_players
        af = self.action_factory
        for p in self.game.players:
            #  A set of default actions for all players
            action_obs: Dict[str, np.ndarray] = af.action_range_to_numpy([])
            assert self.next_accepted_action, "Must have action related"
            if p.id == self.next_player:
                action_obs = af.action_range_to_numpy(self.next_accepted_action)
            obs[p.id] = spaces.flatten(
                # Note this includes action observation
                self.observation_space,
                [action_obs, self.state.to_player_numpy_data(p.id)],
            )
        return obs

    @property
    def observation_space(self) -> spaces.Space:
        if self.cached_space is not None:
            return self.cached_space
        self.cached_space = spaces.Tuple(
            [self.action_factory.action_space_possible, self.state.observation_space,]
        )
        return self.cached_space

    @property
    def reward_range(self) -> Tuple[int, int]:
        low, high = self.game.reward_range
        assert low >= Reward.INVALID_ACTION, "Punish reward must be higher than game"
        return low, high

    # TODO: The return is an instance of the observation
    def reset(self) -> List[np.ndarray]:
        """Return instance of environment
        """
        init_state = self.game.reset()
        self.game_gen = self.game.start()
        self.next_player, self.next_accepted_action, _ = next(self.game_gen)
        self.cached_space = None
        self.continuous_invalid_inputs = []

        return self.__get_all_players_observation_with_action()

    def step(
        self, agents_action: List[int]
    ) -> Tuple[
        List[spaces.Space],  # observations
        List[int],  # rewards
        List[bool],  # terminals
        Dict,  # info
    ]:
        assert self.game_gen, "Must run reset first"

        rewards = [0] * self.n_agents

        # Let's check for the right action
        assert self.next_accepted_action

        action_to_send = None

        for player_id, a in enumerate(agents_action):
            assert isinstance(
                a, np.ndarray
            ), f"Expect action of type nd.array (got: {a.__class__})"
            # Decode value from open_ai
            action = self.action_factory.from_numpy(agents_action[player_id])
            assert isinstance(action, ActionInstance)
            # logging.warning(f"Player {player_id} got action: {action}")
            if player_id == self.next_player:
                if self.action_factory.is_valid_from_range(
                    action, self.next_accepted_action
                ):
                    action_to_send = action
                else:
                    logging.warning(f"🙅‍♂️ Action {action} is not valid.")
                    self.continuous_invalid_inputs.append(action)
                    rewards[player_id] = Reward.INVALID_ACTION

            else:
                # Make sure for other player, the action is appropriate
                # If it is not their turn and they move, punish!
                expected_default_action: ActionInstance = self.action_factory.default
                if action != expected_default_action:
                    rewards[player_id] = Reward.INVALID_ACTION
                else:
                    rewards[player_id] = Reward.VALID_ACTION

        # check if we have a valid action, and return that
        if not action_to_send:
            termination = [False] * self.game.number_of_players
            if (
                len(self.continuous_invalid_inputs)
                >= self.max_continuous_invalid_inputs
            ):
                logging.warning(
                    f"Getting continue bad input: {self.continuous_invalid_inputs}"
                )
                termination = [True] * self.game.number_of_players
            observations = self.__get_all_players_observation_with_action()
            return (observations, rewards, termination, {})
        self.continuous_invalid_inputs = []

        # Send the action to the step
        stopped_iteration = False
        try:
            (next_player, accepted_action, last_reward) = self.game_gen.send(
                action_to_send
            )
        except StopIteration:
            # Game has finished! Return final state
            stopped_iteration = True

        # Let's get observation for each of the player
        observations = self.__get_all_players_observation_with_action()
        if stopped_iteration:
            return (
                observations,
                # TODO: return winner's reward
                [Reward.VALID_ACTION] * self.game.number_of_players,
                # All players to terminate
                [True] * self.game.number_of_players,
                {},
            )

        rewards[self.next_player] = last_reward
        self.next_player = next_player
        self.next_accepted_action = accepted_action
        terminal = [False for _ in range(self.game.number_of_players)]

        return observations, rewards, terminal, {}

    def render(self, mode="human"):
        """Render the relevant cards
        """
        pass

    def to_player_data(self, player_id: int):
        return self.state.to_player_data(player_id)

    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def close(self):
        return


class HumanAgent(Agent):
    """Represent a human agent in the world."""

    env: GameWrapperEnvironment

    def __init__(self, env):
        self.env = env

    def forward(self, observation) -> np.ndarray:
        """Forward a step for the agent

        :return: observation setup
        """
        env = self.env
        assert env.next_player is not None
        pprint(env.to_player_data(env.next_player))
        prompt = "👀 Please enter action ({}):".format(env.next_accepted_action)
        chosen_action = None
        while not chosen_action:
            try:
                given_action = input(prompt)
                chosen_action = env.action_factory.from_str(given_action)
                print(f"😋 Chosen action: {chosen_action}")
            except InvalidActionError as e:
                print("🙅‍♂️ Invalid action.")
                print(str(e))

        # Now from the chosen action, convert back to np.ndarray
        chosen_action_numpy = env.action_factory.to_numpy(chosen_action)
        return chosen_action_numpy


class EnvironmentInteration:
    """Represent a game that can be played"""

    env: GameWrapperEnvironment
    agents: List[Agent]
    episodes: int
    max_same_player: int

    def __init__(self, env, agents, episodes=50, max_same_player=20):
        assert len(agents) == env.n_agents
        self.env = env
        self.agents = agents
        self.episodes = episodes
        self.max_same_player = max_same_player

    def play(self):
        env = self.env
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

                default_numpy_action = env.action_factory.to_numpy(
                    env.action_factory.default
                )
                action_n = [default_numpy_action] * env.n_agents

                # print(f"Player {player_id+1} taking action...")

                action_taken = self.agents[player_id].forward(obs_n[player_id])
                # print(f"Action taken: {action_taken}")
                assert isinstance(
                    action_taken, np.ndarray
                ), "Forward agent should return an ndarray. (Got: {action_taken.__class__})"

                action_n[player_id] = action_taken

                obs_n, reward_n, done_n, _ = env.step(action_n)
                ep_reward += sum(reward_n)
                env.render()

                last_player = player_id

            print("Episode #{} Reward: {}".format(ep_i, ep_reward))
        env.close()
