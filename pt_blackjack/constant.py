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
    min_bet_per_round: int = 1


class Reward(ut.Reward):
    PUNISH_BUSTED = -3
    REWARD_EXACT_POINTS = 3
    REWARD_WINNER = 5
    PUNISH_LOSER = -5
