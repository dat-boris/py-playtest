import numpy as np

import gym.spaces as spaces

from playtest.components.card import Card, BasicDeck as Deck


def test_observation():
    """Ensure that the returned deck fits into openai's observation
    """
    deck1 = Deck(all_cards=True, shuffle=False)
    obs_space = deck1.get_observation_space()
    assert isinstance(obs_space, spaces.Space)

    expected_first_card = Card.from_str("A,S")

    deck_data = deck1.to_data()
    assert deck_data[0] == expected_first_card.to_data()

    deck1_data = deck1.to_data_for_numpy()
    assert len(deck1_data) == 52, "Each card represented by 2 integer"
    first_card = deck1_data[0]
    assert first_card == expected_first_card.to_data_for_numpy()


class PartialDeck(Deck):
    @staticmethod
    def get_max_size() -> int:
        return 3


def test_partial_hand():
    """Given a partial hand, ensure that we are able to map this to
    a partial deck and does not return a whole array.

    This is the behavior when data and numpy are difference:

    * data - just return a list of the the data points
    * numpy array - return a fixed numpy array of max size
    """
    deck_empty = PartialDeck(cards=[])
    assert deck_empty.to_data() == []
    assert deck_empty.to_data_for_numpy() == (
        [Card.get_null_data()] * PartialDeck.get_max_size()
    )

    expected_first_card = Card.from_str("A,S")
    deck = PartialDeck(cards=[expected_first_card] * 2)
    assert deck.to_data() == [expected_first_card.to_data()] * 2

    # Test for flatten numpy data
    numpy_data = deck.to_data_for_numpy()
    assert len(numpy_data) == PartialDeck.get_max_size()
    obs_space = deck.get_observation_space()
    flattend_numpy = spaces.flatten(obs_space, numpy_data)
    assert (flattend_numpy[0:2] == expected_first_card.to_data_for_numpy()).all()
    assert (flattend_numpy[2:4] == expected_first_card.to_data_for_numpy()).all()
    assert (flattend_numpy[4:6] == Card.get_null_data()).all()


def test_deck_deal():
    deck1 = Deck(all_cards=True, shuffle=False)
    deck2 = Deck([])

    deck1.deal(deck2, count=2)
    expected = Deck([Card.from_str(c) for c in ["K,C", "K,D"]]).to_data()
    deck2_data = deck2.to_data()
    assert deck2_data == expected


def test_deck_value():
    deck = Deck([Card.from_str(c) for c in ["T,D", "A,C"]])
    assert sum([c.number for c in deck]) == 11


def test_reset():
    deck = Deck(all_cards=True)
    assert len(deck) == 52
    deck.reset()
    assert len(deck) == 52

    deck = Deck([Card.from_str(c) for c in ["A,D", "Q,S"]])
    assert len(deck) == 2
    deck.reset()
    assert deck[0] == Card.from_str("A,D")
