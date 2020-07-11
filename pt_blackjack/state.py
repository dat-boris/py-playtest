from typing import List

from playtest.components import BasicDeck, Token
from playtest import SubState, FullState, Visibility


class PlayerState(SubState):

    visibility = {
        "hand": Visibility.SELF,
        "bank": Visibility.ALL,
        "bet": Visibility.ALL,
    }

    hand: BasicDeck
    bank: Token
    bet: Token

    def __init__(self, param=None):
        self.hand = BasicDeck([])  # max=5, visibility='owner'
        self.bank = Token([param.starting_pot] if param else [0])
        self.bet = Token([0])  # visibility='all'


class State(FullState):

    visibility = {
        "deck": Visibility.NONE,
        "discarded": Visibility.ALL,
        "current_player": Visibility.ALL,
        "number_of_rounds": Visibility.ALL,
        "hit_rounds": Visibility.ALL,
    }

    player_state_class = PlayerState

    deck: BasicDeck
    discarded: BasicDeck
    players: List[PlayerState]
    current_player: int
    number_of_rounds: int
    hit_rounds: int

    def __init__(self, param=None):
        super().__init__(param=param)
        self.deck = BasicDeck(all_cards=True, shuffle=True)
        self.discarded = BasicDeck([])
        self.current_player = 0
        self.number_of_rounds = 0
        self.hit_rounds = 0
