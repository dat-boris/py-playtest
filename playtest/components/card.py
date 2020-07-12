import abc
import enum
import logging
import random
from copy import copy
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

    value: List[enum.IntEnum]
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
                all_cards.append(cls([i, suite]))
        return all_cards

    @classmethod
    def get_observation_space(cls) -> spaces.Space:
        return spaces.Box(
            low=0, high=max(len(CardNumber), len(CardSuite)), shape=(2,), dtype=np.int8
        )


C = TypeVar("C", bound=BaseCard)


# TODO: Refactor this as a more generic component
class Deck(Component, Generic[C]):
    """Represent a list of Cards
    """

    # This can be overridden to a specific Card type.
    # Note that the card must implement the method above.
    generic_card: Type[C]

    value: List[C]

    init_value: List[C]
    shuffle: bool

    @classmethod
    @abc.abstractmethod
    def get_max_size(cls) -> int:
        """Return the maximum size of this class

        e.g. For a deck of cards will be 52
        """
        raise NotImplementedError()

    def __str__(self):
        return "{}...".format(str(self.value[5:]))

    def __init__(self, cards=None, shuffle=False, all_cards=False):
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

        self.init_value = copy(cards)
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        # We use init_value to ensure that when we reset, we still
        # keep the same set of cards ready
        self.value = copy(self.init_value)

    def deal(self, other: "Deck", count=1, all=False):
        """Deal cards to another deck"""
        if all:
            count = len(self)
        for _ in range(count):
            assert self.value, f"Oops - Deck {self.__class__} ran out of card."
            other.add(self.value.pop())

    def pop(self, index=-1, count=1, all=False) -> List[C]:
        if all:
            count = len(self)
        cards_popped = []
        for _ in range(count):
            cards_popped.append(self.value.pop(index))
        return cards_popped

    def move_to(self, other: "Deck", card: C):
        """Move a specific card to other deck"""
        assert isinstance(other, self.__class__)
        self.value.remove(card)
        other.add(card)

    def add(self, card: C):
        self.value.append(card)

    def remove(self, card: C):
        self.value.remove(card)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i):
        assert isinstance(i, int), "Deck only takes integer subscription"
        return self.value[i]

    def __iter__(self):
        return iter(self.value)

    def to_data(self):
        return [c.to_data() for c in self.value]

    @classmethod
    def from_data(cls, data):
        cards = []
        for card_data in data:
            cards.append(cls.generic_card.from_data(card_data))
        return cls(cards)

    def to_data_for_numpy(self):
        value_array = self.to_data()
        empty_slot_count = self.get_max_size() - len(value_array)
        assert empty_slot_count >= 0, f"Deck have too many cards '{self.value}'"
        if empty_slot_count >= 0:
            value_array += [self.generic_card.get_null_data()] * empty_slot_count
        assert len(value_array) == self.get_max_size()
        return value_array

    @classmethod
    def get_observation_space(cls) -> spaces.Space:
        """Return the observation space of this space
        """
        card_obs_space = cls.generic_card.get_observation_space()
        return spaces.Tuple([card_obs_space] * cls.get_max_size())

    def __get_null_card_data(self) -> List[int]:
        """Return the null data object for the card
        """
        card_len = spaces.flatdim(self.generic_card.get_observation_space())
        return [0] * card_len


class BasicDeck(Deck[Card]):
    generic_card = Card

    value_type = [Card] * 52

    @staticmethod
    def get_max_size() -> int:
        return 52
