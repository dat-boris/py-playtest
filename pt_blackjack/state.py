from typing import List

from playtest.components import Deck, Token
from playtest import SubState, FullState


class PlayerState(SubState):

    full_data_spec = {
        "hand": Deck,
        "bank": Token,
        "bet": Token,
    }

    visible_data_spec = {
        "bank": Token,
        "bet": Token,
    }

    hand: Deck
    bank: Token
    bet: Token

    def __init__(self, param=None):
        self.hand = Deck([])  # max=5, visibility='owner'
        self.bank = Token(param.starting_pot if param else 0)
        self.bet = Token(0)  # visibility='all'


class State(FullState):

    full_data_spec = {
        "deck": Deck,
        "discarded": Deck,
    }

    visible_data_spec = {
        "discarded": Deck,
    }

    player_state_class = PlayerState

    deck: Deck
    discarded: Deck
    players: List[PlayerState]

    def __init__(self, param=None):
        super().__init__(param=param)
        self.deck = Deck(all_cards=True, shuffle=True)
        self.discarded = Deck([])

    def get_player_state(self, player_id: int) -> PlayerState:
        assert isinstance(player_id, int)
        return self.players[player_id]
