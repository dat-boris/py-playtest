from dataclasses import dataclass

import playtest.constant as ut


@dataclass
class Param(ut.Param):
    number_of_players: int = 2
    starting_pot: int = 10
    max_score: int = 21
    number_of_rounds: int = 3
    max_hits: int = 3
    max_bank: int = 20


class Reward(ut.Reward):
    BETTED = 3
    HITTED = 5
    SKIPPED = 5
