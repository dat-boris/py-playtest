from typing import Any, Set, List, Tuple
import random
import re

import numpy as np
import gym.spaces as spaces

from playtest.action import ActionRange, ActionInstance, ActionEnum


class ActionIntInSet(ActionRange):

    action_name: ActionEnum
    # A list of values mapped to position
    valid_range: List[int]

    def __init__(self, action_name: ActionEnum, valid_range: Set[int]):
        # For boolean state, no need to do things
        assert isinstance(action_name, ActionEnum)
        super().__init__(action_name, list(sorted(valid_range)))

    def is_legal(self, x: ActionInstance, legal_range) -> bool:
        return x.key == self.action_name and x.value in legal_range

    def pick_random(self, legal_range: Any) -> ActionInstance:
        picked_value = random.choice(list(legal_range))
        return ActionInstance(key=self.action_name, value=picked_value)

    # ---------
    # Int marshalling - for openAI gym interaction
    # ---------

    def get_number_of_distinct_value(self) -> int:
        return len(self.valid_range)

    def to_int(self, value: int) -> int:
        assert isinstance(value, int)
        return value

    def from_int(self, np_value: int) -> ActionInstance:
        """Check if value is acceptable"""
        mapped_value = self.valid_range[np_value]
        return ActionInstance(key=self.action_name, value=mapped_value)

    # ---------
    # Str marshalling - for human interaction
    # ---------

    def from_str(self, action_str: str) -> ActionInstance:
        action_str = action_str.lower()
        action_key = self.action_name.value.lower()
        matches = re.match(f"{action_key}[(](\\d+)[)]", action_str)
        assert matches
        instance_value = int(matches.group(1))
        return ActionInstance(key=self.action_name, value=instance_value)

    # ---------
    # Gym space marshalling - for showing what action is available
    # ---------

    def get_action_space_possible(self):
        return spaces.MultiBinary(self.get_number_of_distinct_value())

    def to_numpy_data(self, legal_range) -> np.ndarray:
        """Return action space possible in numpy array
        """
        array_value = [0] * self.get_number_of_distinct_value()
        assert len(array_value) == len(
            self.valid_range
        ), "Valid range should be same len"
        for i, v in enumerate(self.valid_range):
            if v in legal_range:
                array_value[i] = 1
        return array_value

    def to_numpy_empty_action(self) -> np.ndarray:
        """Return null data set for ActionRange
        """
        array_value = [0] * self.get_number_of_distinct_value()
        return array_value


class ActionIntInRange(ActionRange):

    action_name: ActionEnum
    valid_range: Tuple[int, int]

    def __init__(self, action_name: ActionEnum, valid_range: Tuple[int, int]):
        # For boolean state, no need to do things
        assert isinstance(action_name, ActionEnum)
        super().__init__(action_name, valid_range)
        assert self.valid_range[0] < self.valid_range[1]

    def __value_list(self) -> List[int]:
        return list(range(self.valid_range[0], self.valid_range[1]))

    def is_legal(self, x: ActionInstance, legal_range) -> bool:
        lower_bound, upper_bound = legal_range
        return x.key == self.action_name and lower_bound <= x.value < upper_bound

    def pick_random(self, legal_range: Any) -> ActionInstance:
        lower, higher = legal_range
        picked_value = random.choice(list(range(lower, higher)))
        return ActionInstance(key=self.action_name, value=picked_value)

    # ---------
    # Int marshalling - for openAI gym interaction
    # ---------

    def get_number_of_distinct_value(self) -> int:
        lower_bound, upper_bound = self.valid_range
        return upper_bound - lower_bound

    def to_int(self, value: int) -> int:
        assert isinstance(value, int)
        return value

    def from_int(self, np_value: int) -> ActionInstance:
        """Check if value is acceptable"""
        lower_bound, _ = self.valid_range
        return ActionInstance(key=self.action_name, value=lower_bound + np_value)

    # ---------
    # Str marshalling - for human interaction
    # ---------

    def from_str(self, action_str: str) -> ActionInstance:
        action_str = action_str.lower()
        action_key = self.action_name.value.lower()
        matches = re.match(f"{action_key}[(](\\d+)[)]", action_str)
        assert matches
        instance_value = int(matches.group(1))
        return ActionInstance(key=self.action_name, value=instance_value)

    # ---------
    # Gym space marshalling - for showing what action is available
    # ---------

    def get_action_space_possible(self):
        lower_bound, upper_bound = self.valid_range
        return spaces.Box(low=lower_bound, high=upper_bound, shape=(2,), dtype=np.int8,)

    def to_numpy_data(self, legal_range) -> np.ndarray:
        """Return action space possible in numpy array
        """
        lower_bound, upper_bound = legal_range
        return [lower_bound, upper_bound]

    def to_numpy_empty_action(self) -> np.ndarray:
        """Return null data set for ActionRange
        """
        return [0, 0]
