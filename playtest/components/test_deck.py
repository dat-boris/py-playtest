from playtest.components.card import Card, BasicDeck as Deck


def test_deck_deal():
    deck1 = Deck(all_cards=True, shuffle=False)
    deck2 = Deck([])

    deck1.deal(deck2, count=2)
    expected = Deck([Card(c) for c in ["Kc", "Kd"]]).to_data()
    assert deck2.to_data() == expected


def test_deck_value():
    deck = Deck([Card(c) for c in ["Tc", "Ac"]])
    assert sum([c.number for c in deck]) == 11


def test_reset():
    deck = Deck(all_cards=True)
    assert len(deck) == 52
    deck.reset()
    assert len(deck) == 52

    deck = Deck([Card(c) for c in ["Ad", "Qs"]])
    assert len(deck) == 2
    deck.reset()
    assert deck[0] == Card("Ad")
