import logging
import random
from copy import copy
from typing import List, Type, Sequence, Optional, Dict, Generic, TypeVar

import numpy as np
import gym.spaces as spaces

from .core import Component


class BaseCard(Component):
    """Represent a baseCard to be implemented

    The most important note is the value of the string,
    which allows
    """

    # A serial id which is used to map to a value
    uid: Optional[int]
    # A string that represent the concise description of the card
    value: str
    # A test wartermark used to test cards being moved in tests
    test_watermark: Optional[str]

    # Represent the total number of unique cards
    # Default to something large
    total_unique_cards: int = 512

    def __init__(self, value: str, uid=None, test_watermark=None):
        assert isinstance(value, str)
        # Ensure that value is stored in a canonical format
        self.value = self.struct_to_value(self.value_to_struct(value))
        self.uid = uid
        self.test_watermark = test_watermark

    def __eq__(self, x):
        return (
            self.value == x.value
            and self.uid == x.uid
            and self.test_watermark == x.test_watermark
        )

    def __repr__(self):
        return self.value

    def reset(self):
        assert False, "Reset should be handled by Deck"

    @classmethod
    def struct_to_value(cls, struct) -> str:
        """A method for converting structure value to the value string
        """
        assert isinstance(struct, str)
        return struct

    @classmethod
    def value_to_struct(cls, value: str):
        """A method for converting the value to structure
        """
        return value

    @staticmethod
    def get_all_cards() -> Sequence["BaseCard"]:
        raise NotImplementedError()

    def to_data(self):
        return self.value + (f" [{self.uid}]" if self.uid is not None else "")

    def to_numpy_data(self) -> int:
        # logging.warn(f"class {self.__class__} has not implemented to_numpy_data")
        return 0


class Card(BaseCard):

    all_suites = ["s", "h", "d", "c"]

    value: str
    # Represent the total number of unique cards
    total_unique_cards: int = 52

    @property
    def suite(self):
        return self.value[1]

    @property
    def number(self) -> int:
        number = self.value[0]
        try:
            return int(number)
        except ValueError:
            return {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 1}[number]

    def is_suite(self, suite: str):
        return self.suite in suite

    @staticmethod
    def get_all_cards(suites=all_suites):
        all_cards = []
        for i in list(range(2, 10)) + ["T", "A", "J", "Q", "K"]:
            for suite in suites:
                all_cards.append(Card(f"{i}{suite}"))
        return all_cards

    def to_numpy_data(self) -> int:
        # ensure that this value is not zero (np.uint8 use zero as null)
        return 1 + self.number * 4 + Card.all_suites.index(self.suite)


C = TypeVar("C", bound=Card)


class Deck(Component, Generic[C]):

    # This can be overridden to a specific Card type.
    # Note that the card must implement the method above.
    generic_card: Type[C]

    cards: List[C]
    init_cards: List[C]
    shuffle: bool
    max_size: int

    def __str__(self):
        return "{}...".format(str(self.cards[5:]))

    def __init__(self, cards=None, shuffle=False, all_cards=False, max_size=52):
        assert getattr(
            self, "generic_card", None
        ), f"Class {self.__class__} have not specificed generic_card"
        if all_cards:
            cards = self.generic_card.get_all_cards()
            assert cards is not None, "Ensure we have cards!"
            if shuffle:
                random.shuffle(cards)
        else:
            assert (
                cards is not None
            ), "Must pass in cards parameter if not specified all_cards"
            cards = [
                self.generic_card(c) if not isinstance(c, self.generic_card) else c
                for c in cards
            ]

        self.init_cards = copy(cards)
        self.max_size = max_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        # We use init_cards to ensure that when we reset, we still
        # keep the same set of cards ready
        self.cards = copy(self.init_cards)

    def to_data(self):
        return [c.to_data() for c in self.cards]

    def deal(self, other: "Deck", count=1, all=False):
        """Deal cards to another deck"""
        if all:
            count = len(self)
        for _ in range(count):
            assert self.cards, f"Oops - Deck {self.__class__} ran out of card."
            other.cards.append(self.cards.pop())

    def pop(self, count=1, all=False) -> List[C]:
        if all:
            count = len(self)
        cards_popped = []
        for _ in range(count):
            cards_popped.append(self.cards.pop())
        return cards_popped

    def move_to(self, other: "Deck", card: C):
        """Move a specific card to other deck"""
        assert isinstance(other, self.__class__)
        self.cards.remove(card)
        other.cards.append(card)

    def add(self, card: C):
        self.cards.append(card)

    def remove(self, card: C):
        self.cards.remove(card)

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, i):
        assert isinstance(i, int), "Deck only takes integer subscription"
        return self.cards[i]

    def __iter__(self):
        return iter(self.cards)

    def get_observation_space(self) -> spaces.Space:
        # Represent a deck of 52 cards
        return spaces.Box(
            low=0,
            high=self.generic_card.total_unique_cards,
            shape=(self.max_size,),
            dtype=np.uint8,
        )

    def to_numpy_data(self) -> np.ndarray:
        value_array = [c.to_numpy_data() for c in self.cards]
        assert len(value_array) <= self.max_size
        return np.array(
            value_array + [0] * (self.max_size - len(value_array)), dtype=np.uint8
        )


class BasicDeck(Deck):
    generic_card = Card
