import enum
import logging
import random
from copy import copy
import abc
from typing import List, Type, Sequence, Optional, Dict, Generic, TypeVar, Union, Tuple

import numpy as np
import gym.spaces as spaces

from .core import Component


class BaseCard(Component):
    """Represent a baseCard to be implemented

    The value can be any composite types which can be flattern
    """

    # A test wartermark used to test cards being moved in tests
    test_watermark: Optional[str]

    def __init__(self, value, param=None, test_watermark=None):
        self.test_watermark = test_watermark
        super().__init__(value, param=param)

    @classmethod
    def get_all_cards(cls) -> Sequence["BaseCard"]:
        raise NotImplementedError()


class CardSuite(enum.IntEnum):
    """One character representing the suite
    """

    S = 1
    H = 2
    D = 3
    C = 4


class CardNumber(enum.IntEnum):
    A = 1
    _2 = 2
    _3 = 3
    _4 = 4
    _5 = 5
    _6 = 6
    _7 = 7
    _8 = 8
    _9 = 9
    T = 10
    J = 11
    Q = 12
    K = 13


class Card(BaseCard):

    value: Tuple[CardNumber, CardSuite]
    value_type = (CardNumber, CardSuite)

    @property
    def suite(self):
        return self.value[1]

    @property
    def number(self) -> int:
        return int(self.value[0])

    @classmethod
    def get_all_cards(cls):
        all_cards = []
        for i in CardNumber:
            for suite in CardSuite:
                all_cards.append(cls((i, suite)))
        return all_cards

    @classmethod
    def get_observation_space(cls) -> spaces.Space:
        return spaces.Box(
            low=0, high=max(len(CardNumber), len(CardSuite)), shape=(2,), dtype=np.int8
        )


C = TypeVar("C", bound=BaseCard)


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
            other.add(self.cards.pop())

    def pop(self, index=-1, count=1, all=False) -> List[C]:
        if all:
            count = len(self)
        cards_popped = []
        for _ in range(count):
            cards_popped.append(self.cards.pop(index))
        return cards_popped

    def move_to(self, other: "Deck", card: C):
        """Move a specific card to other deck"""
        assert isinstance(other, self.__class__)
        self.cards.remove(card)
        other.add(card)

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

    @classmethod
    def get_observation_space(cls) -> spaces.Space:
        # Represent a deck of 52 cards
        return spaces.Box(
            low=0,
            high=cls.generic_card.total_unique_cards,
            shape=(cls.max_size,),
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
