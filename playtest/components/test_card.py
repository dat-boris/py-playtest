from .card import Deck


def test_card_deal():
    deck1 = Deck(all_cards=True, shuffle=False)
    deck2 = Deck([])

    deck1.deal(deck2, count=2)
    expected = Deck(["Kc", "Kd"]).to_data()
    assert deck2.to_data() == expected


def test_card_value():
    deck = Deck(["Tc", "Ac"])
    assert sum([c.number for c in deck]) == 11


def test_reset():
    deck = Deck(all_cards=True)
    assert len(deck) == 52
    deck.reset()
    assert len(deck) == 52

    deck = Deck(["Ad", "Qs"])
    assert len(deck) == 2
    deck.reset()
    assert deck[0] == "Ad"
