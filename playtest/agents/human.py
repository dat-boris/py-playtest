"""A human agent

Playing against a human agent
"""
from pprint import pprint
import numpy as np
from typing import List

from playtest.action import InvalidActionError
from playtest.agents.base import BaseAgent


class HumanAgent(BaseAgent):
    """Represent a human agent in the world."""

    def get_input(self, prompt: str) -> str:
        """Getting input from environment.

        This can be mocked
        """
        return input(prompt)

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
                given_action = self.get_input(prompt)
                chosen_action = env.action_factory.from_str(given_action)
                print(f"😋 Chosen action: {chosen_action}")
            except InvalidActionError as e:
                print("🙅‍♂️ Invalid action.")
                print(str(e))

        # Now from the chosen action, convert back to np.ndarray
        chosen_action_numpy = env.action_factory.to_numpy(chosen_action)
        return chosen_action_numpy
