import numpy as np

import gym.spaces as spaces

from playtest.components.card import Card, BasicDeck as Deck


def test_observation():
    """Ensure that the returned deck fits into openai's observation
    """
    deck1 = Deck(all_cards=True, shuffle=False)
    obs_space = deck1.observation_space
    assert isinstance(obs_space, spaces.Space)

    expected_first_card = Card.from_str("A,S")

    deck_data = deck1.to_data()
    assert deck_data[0] == expected_first_card.to_data()

    numpy_data = deck1.to_numpy_data()
    assert len(numpy_data) == 52 * 2, "Each card represented by 2 integer"
    first_card = numpy_data[0:2]
    assert Card.from_data(first_card) == expected_first_card


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
    assert (
        deck_empty.to_numpy_data() == np.array([0, 0] * PartialDeck.get_max_size())
    ).all()

    expected_first_card = Card.from_str("A,S")
    deck = PartialDeck(cards=[expected_first_card] * 2)
    assert deck.to_data() == [expected_first_card.to_data()] * 2
    numpy_data = deck.to_numpy_data()
    assert len(numpy_data) == 2 * PartialDeck.get_max_size()
    assert (numpy_data[0:2] == expected_first_card.to_numpy_data()).all()
    assert (numpy_data[2:4] == expected_first_card.to_numpy_data()).all()
    assert (numpy_data[4:6] == [0, 0]).all()


def test_deck_deal():
    deck1 = Deck(all_cards=True, shuffle=False)
    deck2 = Deck([])

    deck1.deal(deck2, count=2)
    expected = Deck([Card.from_str(c) for c in ["K,C", "K,D"]]).to_data()
    assert deck2.to_data() == expected


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
