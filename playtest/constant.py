class Param:
    number_of_players: int

    def __init__(self, number_of_players):
        self.number_of_players = number_of_players


class Reward:
    """A constant class represent various rewards"""

    DEFAULT = 0
    INVALID_ACTION = -10
    VALID_ACTION = 1
    WINNER = 100
    LOSER = -1
