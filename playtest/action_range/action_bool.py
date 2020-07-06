from typing import Any

import numpy as np

from playtest.action import ActionRange, ActionInstance, ActionEnum


class ActionBooleanRange(ActionRange):

    action_name: ActionEnum
    valid_range: Any

    def __init__(self, action_name: ActionEnum, valid_range: None):
        # For boolean state, no need to do things
        assert isinstance(action_name, ActionEnum)
        super().__init__(action_name, valid_range)

    def is_legal(self, x: ActionInstance, legal_range) -> bool:
        return x.key == self.action_name

    def __get_action(self) -> ActionInstance:
        """Return an instance of action"""
        return ActionInstance(self.action_name, True)

    def pick_random(self) -> ActionInstance:
        return self.__get_action()

    # ---------
    # Int marshalling - for openAI gym interaction
    # ---------

    def get_number_of_distinct_value(self) -> int:
        return 1

    def to_int(self, value: bool) -> int:
        assert value is True
        return 0

    def from_int(self, np_value: int) -> ActionInstance:
        """Check if value is acceptable"""
        # Yes, this is zero (only 1 value avaliable)
        if np_value == 0:
            return self.__get_action()
        raise KeyError(f"Unknown value {np_value} for {self}")

    # ---------
    # Str marshalling - for human interaction
    # ---------

    def from_str(self, action_str: str) -> ActionInstance:
        assert action_str.startswith(self.action_name.value + "(")
        return self.__get_action()

    # ---------
    # Gym space marshalling - for showing what action is available
    # ---------

    def get_action_space_possible(self):
        return spaces.MultiBinary(1)

    def to_numpy_data(self) -> np.ndarray:
        """Return action space possible in numpy array
        """
        return np.array([1])
