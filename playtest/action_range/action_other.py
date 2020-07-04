# TODO: import and fixes these


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
        if not self.minimum_value <= value <= self.maximum_value:
            raise IllegalActionError(
                f"Value {value} not within bound [{self.minimum_value}, {self.maximum_value})"
            )

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
        raise IllegalActionError(f"Unknown action: {action_str}")

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


# TODO: rename this as this phrase is a lot better
ActionInt = ActionSingleValue


ASV = TypeVar("ASV", bound=ActionSingleValue)


class ActionSingleValueRange(ActionRange[ASV, S]):
    """The action can takes a range of value.

    Let's say we have an action that plays a bet

        ActionBet(amount)

    You can bet any amount from 5 (minimum bet!) to your bank amount,
    then you can use this action range.
    """

    instance_class: Type[ASV]

    # Note: upper is inclusive
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

    def pick_random(self) -> ActionInstance:
        return self.instance_class(random.randint(self.lower, self.upper))

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

    def is_legal(self, x) -> bool:
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
            raise IllegalActionError(f"{value} is not one of {self.value_set_mapping}")
        self.value = value

    def __repr__(self):
        return f"{self.key}({self.value})"

    def __eq__(self, x):
        return self.key == x.key and self.value == x.value

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
                instance_value = int(instance_value)  # type: ignore
            return cls(instance_value)
        raise IllegalActionError(f"Unknown action: {action_str}")

    @classmethod
    def from_int(cls, np_value: int) -> ActionInstance:
        assert 0 <= np_value < cls.unique_value_count
        return cls(cls.value_set_mapping[np_value])


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
        legal_value_str = ",".join([str(v) for v in sorted(self.possible_values)])
        return f"{self.instance_class.key}([{legal_value_str}])"

    def __eq__(self, x):
        return (
            self.__class__ == x.__class__ and self.possible_values == x.possible_values
        )

    def pick_random(self) -> ActionInstance:
        return self.instance_class(random.choice(list(self.possible_values)))

    def is_actionable(self):
        return bool(self.possible_values)

    def is_legal(self, action: AIS):
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


# TODO: Rename these
class ActionIntInSet:
    def __init__(self, value):
        pass


class ActionBool:
    def __init__(self):
        pass


class ActionIntInRange:
    def __init__(self, lower, upper):
        pass


class BaseDecision:
    pass
