import pytest
import numpy as np

import gym.spaces as spaces

from .card import Card, CardNumber, CardSuite, BasicDeck


def test_card():
    """Testing counter as a simple conversion
    """
    with pytest.raises(TypeError):
        # Excpecting list of enums
        Card(value=(1, 3))

    c = Card(value=[CardNumber.T, CardSuite.S])
    c2 = Card(value=[CardNumber._9, CardSuite.D])

    data_representation = c.to_data()
    assert data_representation == [10, 1]
    new_obj = Card.from_data(data_representation)
    assert new_obj.value == [10, 1]
    # One more round trip
    assert new_obj.to_data() == [10, 1]

    assert isinstance(c.get_observation_space(), spaces.Box)
    # Note the box conversion convert this to an array
    assert (c.to_data_for_numpy() == np.array([10, 1])).all()

    # Comparison depends on numpy
    assert new_obj == c, "Comparison works with numpy"

    assert repr(c) == "Card(T,S)", "Str representation works"
    assert Card.from_str("T,S") == c
    assert repr(c2) == "Card(_9,D)", "Str representation works"
    assert Card.from_str("_9,D") == c2


def test_deck():
    c = Card(value=[CardNumber.T, CardSuite.S])
    d = BasicDeck([c])

    deck_data = d.to_data()
    assert deck_data == [[10, 1]]
    new_deck = BasicDeck.from_data(deck_data)
    assert new_deck.to_data() == [[10, 1]]
