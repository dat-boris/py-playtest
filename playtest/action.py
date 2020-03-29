import re
import abc
import itertools
from typing import Optional, Type, Sequence, Dict, TypeVar, Generic, Set, List

import numpy as np

import gym.spaces as spaces

from .state import FullState
from .logger import Announcer
from .constant import Param
from .components.core import Component


class InvalidActionError(RuntimeError):
    pass


S = TypeVar("S", bound=FullState)


class ActionInstance(abc.ABC, Generic[S]):
    """Represent an instance of the action.

    The instance of the action can be of acted on.
    """

    key: str

    def __init__(self, value):
        pass

    def __eq__(self, x):
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        """Create string representation"""
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    @classmethod
    @abc.abstractmethod
    def from_str(cls, action_str: str) -> "ActionInstance":
        raise NotImplementedError(f"Action {cls} was not implemented")

    @classmethod
    @abc.abstractmethod
    def get_number_of_distinct_value(cls) -> int:
        """Return the max possible value for the action

        Note this is inclusive.
        e.g. if action can take range of [0,1] -> max_value = 2
        """
        raise NotImplementedError(f"Action {cls} was not implemented")

    @abc.abstractmethod
    def to_int(self) -> int:
        raise NotImplementedError(f"Action {self.__class__} was not implemented")

    @classmethod
    @abc.abstractmethod
    def from_int(cls, np_value: int) -> "ActionInstance":
        raise NotImplementedError(f"Action {cls} was not implemented")

    @abc.abstractmethod
    def resolve(
        self, s: S, player_id: int, a: Optional[Announcer] = None
    ) -> Optional["ActionRange"]:
        """This resolves the action

        :return:
            Return None if this action is complete resolved.
            Can also return additional action range.
        """
        raise NotImplementedError()


AI = TypeVar("AI", bound=ActionInstance)


class ActionRange(abc.ABC, Generic[AI, S]):
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

    @abc.abstractmethod
    def __init__(self, state: S, player_id: int):
        self.player_id = player_id
        self.actionable = True

    def __str__(self):
        return repr(self)

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError(f"{self.__class__} is not implemented")

    @abc.abstractmethod
    def __eq__(self, x):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_action_space_possible(cls) -> spaces.Space:
        raise NotImplementedError(f"{cls} is not implemented")

    @abc.abstractmethod
    def to_numpy_data(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__} is not implemented")

    @abc.abstractmethod
    def is_actionable(self) -> bool:
        """Return if this action is actionable"""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_valid(self, x: AI) -> bool:
        raise NotImplementedError()


class ActionBoolean(ActionInstance[S]):
    """Represent a boolean action"""

    # Taking value to satisfy ActionInstance type
    def __init__(self, value=True):
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
    def get_number_of_distinct_value(cls) -> int:
        return 2

    def to_int(self) -> int:
        return 1

    @classmethod
    def from_int(cls, np_value: int) -> "ActionInstance":
        """Check if value is acceptable"""
        if np_value == 1:
            return cls()
        raise InvalidActionError(f"Unknown value {np_value} for {cls}")


class ActionBooleanRange(ActionRange[AI, S]):
    def __init__(self, state: S, player_id: int):
        # For boolean state, no need to do things
        self.actionable = True

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
    # maximum_value is inclusive
    maximum_value: int

    def __init__(self, value: int):
        self.value = value
        assert self.maximum_value is not None, "{self.__class__} must set max_value"
        assert self.minimum_value is not None, "{self.__class__} must set min_value"

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
    def get_number_of_distinct_value(cls):
        return cls.maximum_value - cls.minimum_value + 1

    def to_int(self) -> int:
        return self.value - self.minimum_value

    @classmethod
    def from_int(cls, np_value: int) -> ActionInstance:
        value = np_value + cls.minimum_value
        assert cls.minimum_value <= value <= cls.maximum_value
        return cls(value)


ASV = TypeVar("ASV", bound=ActionSingleValue)


class ActionSingleValueRange(ActionRange[ASV, S]):
    """The action can takes a range of value.

    Let's say we have an action that plays a bet

        ActionBet(amount)

    You can bet any amount from 5 (minimum bet!) to your bank amount,
    then you can use this action range.
    """

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


T = TypeVar("T")


class ActionValueInSet(ActionInstance[S], Generic[S, T]):

    value: T
    # A mapping from the value into specific position
    value_set_mapping: List[T]
    unique_value_count: int
    # Set this if coercing string into in when from_str
    coerce_int = False

    def __init__(self, value):
        if value not in self.value_set_mapping:
            raise InvalidActionError(f"{value} is not one of {self.value_set_mapping}")
        self.value = value

    @classmethod
    def value_to_int(cls, value) -> int:
        return cls.value_set_mapping.index(value)

    @classmethod
    def get_number_of_distinct_value(cls) -> int:
        return cls.unique_value_count

    def to_int(self) -> int:
        return self.value_to_int(self.value)

    @classmethod
    def from_str(cls, action_str: str) -> ActionInstance:
        action_key = cls.key
        matches = re.match(f"{action_key}[(](\\w+)[)]", action_str)
        if matches:
            instance_value = matches.group(1)
            if cls.coerce_int:
                instance_value = int(instance_value)
            return cls(instance_value)
        raise InvalidActionError(f"Unknown action: {action_str}")

    @classmethod
    def from_int(cls, np_value: int) -> ActionInstance:
        assert 0 <= np_value < cls.unique_value_count
        return cls(np_value)


AIS = TypeVar("AIS", bound=ActionValueInSet)


class ActionValueInSetRange(ActionRange[AIS, S], Generic[AIS, S, T]):
    """The action can takes a set of value.

    Let's say we have an action that plays a card from hand:

        ActionPlay(position)

    But you can only play card [1,3,5] in hand.  You will be initialized with

        ActionPlayRange([1,3,5])

    Note that this is comparison with a ActionValueRange action, where you
    takes a range (e.g. a Bet can take value in a range).

    The main difference is the representation of the possible space.  e.g.
    that ActionValueRange is like a Bet.
    """

    instance_class: Type[AIS]
    possible_values: Set[T]

    def __init__(self, state: S, player_id: int):
        raise NotImplementedError()

    def __repr__(self):
        if not self.possible_values:
            return ""
        valid_value_str = ",".join([str(v) for v in sorted(self.possible_values)])
        return f"{self.instance_class.key}([{valid_value_str}])"

    def __eq__(self, x):
        return (
            self.__class__ == x.__class__ and self.possible_values == x.possible_values
        )

    def is_actionable(self):
        return bool(self.possible_values)

    def is_valid(self, action: AIS):
        return action.value in self.possible_values

    @classmethod
    def get_action_space_possible(cls):
        """Return two value, represent
        (high, low)
        """
        return spaces.MultiBinary(cls.instance_class.unique_value_count)

    def to_numpy_data(self) -> np.ndarray:
        array_value = [0] * self.instance_class.unique_value_count
        for v in self.possible_values:
            array_value[self.instance_class.value_to_int(v)] = 1
        return np.array(array_value)

    @classmethod
    def to_numpy_data_null(self) -> np.ndarray:
        return np.array([0] * self.instance_class.unique_value_count)


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
        accepted_range: Optional[Sequence[Type[ActionRange]]] = None,
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

    def get_action_map(self) -> Sequence[ActionRange]:
        """return an array of action range and it's map

        For example, if we have too boolean action:

        The first two int range will belongs to the int
        [actionBoolARange, actionBoolARange, actionBoolBRange, actionBoolBRange]
        """
        action_map = itertools.chain(
            *[
                [action_range]
                * (spaces.flatdim(action_range.get_action_space_possible()) + 1)
                for action_range in self.range_classes
            ]
        )
        return list(action_map)  # type: ignore

    def to_numpy(self, action: ActionInstance) -> np.int64:
        """Converting an action instance to numpy."""
        int_to_return = 0
        found_action = False
        for action_range in self.range_classes:
            action_expected = action_range.instance_class
            action_space_possible = action_range.get_action_space_possible()
            action_int_required = spaces.flatdim(action_space_possible) + 1
            if action.__class__ is action_expected:
                np_value = spaces.flatten(
                    action_expected.get_action_space(), action.to_numpy_data()
                )
                if isinstance(np_value, np.int):
                    assert np_value < action_int_required
                    int_to_return += np_value
                elif isinstance(np_value, np.ndarray):
                    assert len(np_value) == 1, f"Must be one dim array {action}"
                    assert (
                        np_value[0] < action_int_required
                    ), f"{action_range}:{np_value} does not fit into space {action_space_possible} -> int({action_int_required})"
                    int_to_return += np_value[0]
                found_action = True
            else:
                int_to_return += action_int_required

        assert found_action, f"Must contain at least one suitable action: {action}"
        assert isinstance(int_to_return, np.int64)
        return int_to_return

    def from_numpy(self, numpy_input: np.int64) -> ActionInstance:
        """Converting from numpy to an action instance."""
        int_space_searched = 0
        found_action = None
        for action_range in self.range_classes:
            action_expected = action_range.instance_class
            action_space_possible = action_range.get_action_space_possible()
            action_int_required = spaces.flatdim(action_space_possible) + 1
            print(
                f"Eval range: for {action_range}, {int_space_searched} -> {action_int_required}"
            )
            if (
                int_space_searched
                <= numpy_input
                < (int_space_searched + action_int_required)
            ):
                action_space = action_expected.get_action_space()
                if isinstance(action_space, spaces.MultiBinary):
                    return action_expected.from_numpy(
                        np.array([numpy_input - int_space_searched])
                    )
                else:
                    raise NotImplementedError(
                        f"Cannot convert in into type {action_space}"
                    )
            else:
                int_space_searched += action_int_required
        raise InvalidActionError(f"Invalid action input: {numpy_input}.")
