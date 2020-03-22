import re
from typing import Optional, Type, Sequence, Dict, TypeVar, Generic, Set

import numpy as np

import gym.spaces as spaces

from .state import FullState
from .logger import Announcer
from .constant import Param
from .components.core import Component


class InvalidActionError(RuntimeError):
    pass


S = TypeVar("S", bound=FullState)


class ActionInstance(Generic[S]):
    """Represent an instance of the action.

    The instance of the action can be of acted on.
    """

    key: str

    def __init__(self):
        pass

    def __eq__(self, x):
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        """Create string representation"""
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    @classmethod
    def from_str(cls, action_str: str) -> "ActionInstance":
        raise NotImplementedError(f"Action {cls} was not implemented")

    @classmethod
    def get_action_space(cls) -> spaces.Space:
        raise NotImplementedError(f"Action {cls} was not implemented")

    def to_numpy_data(self) -> np.ndarray:
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        """Provide a default numpy representation if this action is not taken
        """
        raise NotImplementedError()

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "ActionInstance":
        raise NotImplementedError(f"Action {cls} was not implemented")

    def resolve(self, s: S, player_id: int, a: Optional[Announcer] = None):
        raise NotImplementedError()


AI = TypeVar("AI", bound=ActionInstance)


class ActionRange(Generic[AI, S]):
    """Represent a range of action

    The action range is responsible for representing a set of potential
    actions, and checking if an action is valid, based on the state
    provided for a specific player.

    Note that the range of action, might not be finite.  For example in a betting
    game, you can bet upwards to your bank amount, which could be infinite.
    """

    instance_class: Type[AI]
    actionable: bool

    player_id: int

    def __init__(self, state: S, player_id: int):
        self.player_id = player_id
        self.actionable = True

    def __str__(self):
        return repr(self)

    def __repr__(self):
        raise NotImplementedError(f"{self.__class__} is not implemented")

    def __eq__(self, x):
        raise NotImplementedError()

    @classmethod
    def get_action_space_possible(cls) -> spaces.Space:
        raise NotImplementedError(f"{cls} is not implemented")

    def to_numpy_data(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__} is not implemented")

    @classmethod
    def to_numpy_data_null(cls) -> np.ndarray:
        raise NotImplementedError(f"{cls} is not implemented.")

    def is_actionable(self) -> bool:
        """Return if this action is actionable"""
        raise NotImplementedError()

    def is_valid(self, x: AI) -> bool:
        raise NotImplementedError()


class ActionBoolean(ActionInstance[S]):
    """Represent a boolean action"""

    def __init__(self):
        # Only need overriding if there's parameter
        pass

    def __eq__(self, x):
        return isinstance(x, self.__class__)

    def __repr__(self) -> str:
        """Create string representation"""
        return self.key

    @classmethod
    def from_str(cls, action_str: str) -> "ActionInstance":
        if action_str == cls.key:
            return cls()
        raise InvalidActionError(f"Unknown action: {action_str}")

    @classmethod
    def get_action_space(cls):
        return spaces.MultiBinary(1)

    def to_numpy_data(self) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([0])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "ActionInstance":
        """Check if value is acceptable"""
        assert len(array) == 1
        if array[0] == 1:
            return cls()
        raise InvalidActionError(f"Unknown value {array} for {cls}")


class ActionBooleanRange(ActionRange[AI, S]):
    def __repr__(self):
        return f"{self.instance_class.key}" if self.actionable else ""

    def __eq__(self, x):
        return self.__class__ == x.__class__

    @classmethod
    def get_action_space_possible(cls):
        return spaces.MultiBinary(1)

    def to_numpy_data(self) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([0])

    def is_actionable(self) -> bool:
        return self.actionable

    def is_valid(self, x: ActionInstance) -> bool:
        return isinstance(x, self.instance_class)


class ActionWait(ActionBoolean[S]):
    key = "wait"

    def resolve(self, state, player_id: int, a=None):
        pass


class ActionWaitRange(ActionBooleanRange[ActionWait, S]):
    instance_class = ActionWait

    actionable = True

    def __init__(self, state, player_id):
        pass


class ActionSingleValue(ActionInstance[S]):
    """A base class for single value action"""

    value: int

    # Define a minimal value for this action
    minimum_value: int
    maximum_value: int

    def __init__(self, value: int):
        self.value = value

    def __repr__(self):
        return f"{self.key}({self.value})"

    def __eq__(self, x):
        return self.key == x.key and self.value == x.value

    @classmethod
    def from_str(cls, action_str: str) -> ActionInstance:
        action_key = cls.key
        matches = re.match(f"{action_key}[(](\\d+)[)]", action_str)
        if matches:
            return cls(int(matches.group(1)))
        raise InvalidActionError(f"Unknown action: {action_str}")

    @classmethod
    def get_action_space(cls):
        return spaces.Box(
            low=cls.minimum_value, high=cls.maximum_value, shape=(1,), dtype=np.int8
        )

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.value])

    @staticmethod
    def to_numpy_data_null() -> np.ndarray:
        return np.array([-1])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> ActionInstance:
        """Check if value is acceptable"""
        assert len(array) == 1
        if array[0] < 0:
            raise InvalidActionError("Must provide positive bet value")
        return cls(int(array[0]))


ASV = TypeVar("ASV", bound=ActionSingleValue)


class ActionSingleValueRange(ActionRange[ASV, S]):
    # Fill in instanceClass here
    instance_class: Type[ASV]

    upper: int
    lower: int
    actionable: bool

    def __repr__(self):
        return f"{self.instance_class.key}({self.lower}->{self.upper})"

    def __eq__(self, x):
        return (
            self.__class__ == x.__class__
            and self.upper == x.upper
            and self.lower == x.lower
        )

    @classmethod
    def get_action_space_possible(cls):
        """Return two value, represent
        (high, low)
        """
        return spaces.Box(
            low=cls.instance_class.minimum_value,
            high=cls.instance_class.maximum_value,
            shape=(2,),
            dtype=np.int8,
        )

    def to_numpy_data(self) -> np.ndarray:
        return np.array([self.lower, self.upper])

    @classmethod
    def to_numpy_data_null(self) -> np.ndarray:
        return np.array([0, 0])

    def is_actionable(self) -> bool:
        return self.actionable

    def is_valid(self, x) -> bool:
        if isinstance(x, self.instance_class):
            return self.lower <= x.value <= self.upper
        return False


AIS = TypeVar("AIS", bound="ActionSingleValue")


class ActionValueInSetRange(ActionRange[AIS, S]):
    values_set: Set[int]

    # Define the maximum number of values possible in set
    max_values_in_set: int

    def __init__(self, state: S, player_id: int):
        raise NotImplementedError()

    def __repr__(self):
        if not self.values_set:
            return ""
        valid_value_str = ",".join([str(v) for v in sorted(self.values_set)])
        return f"{self.instance_class.key}([{valid_value_str}])"

    def is_actionable(self):
        return bool(self.values_set)

    def is_valid(self, action: AIS):
        return action.value in self.values_set

    def value_to_position(self, value) -> int:
        """Converting a value to int"""
        raise NotImplementedError(self.__class__.__name__)

    def position_to_value(self, pos: int):
        raise NotImplementedError(self.__class__.__name__)

    @classmethod
    def get_action_space_possible(cls):
        """Return two value, represent
        (high, low)
        """
        return spaces.MultiBinary(cls.max_values_in_set)

    def to_numpy_data(self) -> np.ndarray:
        array_value = [0] * self.max_values_in_set
        for v in self.values_set:
            array_value[self.value_to_position(v)] = 1
        return np.array(array_value)

    @classmethod
    def to_numpy_data_null(self) -> np.ndarray:
        return np.array([0] * self.max_values_in_set)


class ActionFactory(Generic[S]):

    param: Param
    range_classes: Sequence[Type[ActionRange]]
    # Specify default action for non-active player
    default: ActionInstance = ActionWait()

    def __init__(self, param: Param):
        self.param = param

    def get_actionable_actions(
        self,
        s: S,
        player_id: int,
        accepted_range: Optional[Sequence[Type[ActionRange]]],
    ) -> Sequence[ActionRange]:
        acceptable_action = []
        if accepted_range is None:
            accepted_range = self.range_classes
        for range_class in accepted_range:
            action_range = range_class(s, player_id=player_id)
            if action_range.is_actionable():
                acceptable_action.append(action_range)
        return acceptable_action

    @property
    def action_space(self) -> spaces.Space:
        """Represent the concret space for the action

        See `action_space_possible` for explanation.
        """
        # We get this from the array of action
        action_space_dict: Dict[str, spaces.Space] = {}
        for a in self.range_classes:
            action_key = a.instance_class.key
            # Note that instance class is an object, we need to work with this
            action_space = a.instance_class.get_action_space()
            assert isinstance(
                action_space, spaces.Space
            ), f"{action_key} does not have valid action space"
            action_space_dict[action_key] = action_space
        return spaces.Dict(action_space_dict)

    @property
    def action_space_possible(self) -> spaces.Space:
        """This represent the observed possible action

        For example, for a Bet, you can bet a higher and lower amount, based
        on the bank of the player.

        Let's say a maximum bet range is between (0, 100)

        So:
        Bet.action_space_possible == space.Box(2)
            # ^^^ The possible represent 2 values

        Bet.to_numpy_data_null == [0,100]
            # ^^^^ The value of default action

        Bet.action_space == space.Box(1)
            # ^^^ Represent the one value, that can fall into above

        """
        return spaces.Dict(
            {
                a.instance_class.key: a.get_action_space_possible()
                for a in self.range_classes
            }
        )

    def action_range_to_numpy(
        self, action_possibles: Sequence[ActionRange]
    ) -> Dict[str, np.ndarray]:
        """Based on the list of Action Range, return a list of action possible

        Return: a list of recursive array which can be used for spaces.flatten
        """
        action_possible_dict = {
            a.instance_class.key: a.to_numpy_data_null() for a in self.range_classes
        }
        for a in action_possibles:
            action_key = a.instance_class.key
            assert action_key in action_possible_dict, "Unknown action dict!"
            action_possible_dict[action_key] = a.to_numpy_data()

        return action_possible_dict

    def is_valid_from_range(
        self, action: ActionInstance, action_ranges: Sequence[ActionRange]
    ):
        for action_range in action_ranges:
            if isinstance(
                action, action_range.instance_class
            ) and action_range.is_valid(action):
                return True
        return False

    def from_str(self, action_input: str) -> ActionInstance:
        """Tokenize input from string into ActionInstance"""
        for a in self.range_classes:
            try:
                return a.instance_class.from_str(action_input)
            except InvalidActionError:
                pass
        raise InvalidActionError(f"Unknown action: {action_input}")

    def to_numpy(self, action: ActionInstance) -> np.ndarray:
        """Converting an action instance to numpy."""
        action_dict = {
            a.instance_class.key: a.instance_class.to_numpy_data_null()
            for a in self.range_classes
        }
        found_action = False
        for action_range in self.range_classes:
            action_expected = action_range.instance_class
            if action.__class__ is action_expected:
                action_dict[action_expected.key] = action.to_numpy_data()
                found_action = True
        assert found_action, f"Must contain at least one suitable action: {action}"
        return spaces.flatten(self.action_space, action_dict)

    def from_numpy(self, numpy_input: np.ndarray) -> ActionInstance:
        """Converting from numpy to an action instance."""
        unflattened = spaces.unflatten(self.action_space, numpy_input)
        # Now given the dict, check if any of them are engaged
        err_msgs = []
        for a in self.range_classes:
            numpy_val = unflattened[a.instance_class.key]
            try:
                return a.instance_class.from_numpy(numpy_val)
            except (ValueError, AssertionError, InvalidActionError) as e:
                err_msgs.append(str(e))
        raise InvalidActionError(
            f"Invalid action input: {numpy_input}.  Msg: {err_msgs}"
        )
