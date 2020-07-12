from pprint import pprint
import pytest

from playtest.action import ActionInstance

from .constant import Param
from .state import State, PlayerState
import pt_blackjack.game as gm
import pt_blackjack.action as acn


NUMBER_OF_PLAYERS = 2


def test_start():
    # Arrange
    s = State(Param(number_of_players=NUMBER_OF_PLAYERS))
    assert len(s.players[0].hand) == 0, "First hand is empty"

    # Act
    returned_state, decision, next_state, _ = gm.game_start(s)
    # pprint(s.to_data())

    # Assert - check next game state and what it should be
    assert next_state == gm.GameState.place_bet

    # Assert - check that state makes sense
    assert len(s.players[0].hand) == 2, "First hand is dealed"
    assert len(s.players[1].hand) == 0, "Second hand is not dealed"

    # Assert - ensure decision is within range
    action_space_possible = decision.action_range_to_numpy()
    assert action_space_possible == {
        acn.ActionName.SKIP.name: [0],
        acn.ActionName.HIT.name: [0],
        acn.ActionName.BET.name: [1, 10],
    }


def test_to_next_bet():
    s = State.from_data(
        {
            "current_player": 0,
            "hit_rounds": 0,
            "deck": [
                [8, 2],
                [6, 1],
                [11, 3],
                [10, 4],
                [12, 4],
                [7, 3],
                [5, 1],
                [6, 2],
                [9, 2],
                # ... snipped extra cards
            ],
            "discarded": [],
            "number_of_rounds": 0,
            "players": [
                {"bank": [10], "bet": [0], "hand": [[2, 2], [3, 2]]},
                {"bank": [10], "bet": [0], "hand": []},
            ],
        }
    )

    # Act - make a $3 bet
    bet_action = ActionInstance(acn.ActionName.BET, 3)
    returned_state, decision, next_state, _ = gm.handle_bet(s, action=bet_action)

    # Assert - check the bank, and hit or miss action range
    assert s.players[0].bet == [3], "Correct bet was made"
    assert s.players[0].bank == [7], "Money taken away from bank"
    assert s.players[1].bank == [10]

    action_space_possible = decision.action_range_to_numpy()
    assert action_space_possible == {
        acn.ActionName.SKIP.name: [1],
        acn.ActionName.HIT.name: [1],
        acn.ActionName.BET.name: [0, 0],
    }


def test_hit():
    s = State.from_data(
        {
            "current_player": 0,
            "hit_rounds": 0,
            "deck": [
                [8, 2],
                [6, 1],
                [11, 3],
                # ... snipped extra cards
            ],
            "discarded": [],
            "number_of_rounds": 0,
            "players": [
                {"bank": [7], "bet": [3], "hand": [[2, 2], [3, 2]]},
                {"bank": [10], "bet": [0], "hand": []},
            ],
        }
    )
    assert len(s.players[0].hand) == 2

    # Act - make a hit on the action
    hit_action = ActionInstance(acn.ActionName.HIT, True)
    returned_state, decision, next_state, _ = gm.decide_hit_miss(s, action=hit_action)

    # Assert, ensure that we get another card, and same action
    assert len(s.players[0].hand) == 3, "Player one get another card"

    action_space_possible = decision.action_range_to_numpy()
    assert action_space_possible == {
        acn.ActionName.SKIP.name: [1],
        acn.ActionName.HIT.name: [1],
        acn.ActionName.BET.name: [0, 0],
    }


def test_skip_first_player():
    s = State.from_data(
        {
            "current_player": 0,
            "hit_rounds": 0,
            "deck": [
                [8, 2],
                [6, 1],
                [11, 3],
                # ... snipped extra cards
            ],
            "discarded": [],
            "number_of_rounds": 0,
            "players": [
                {"bank": [7], "bet": [3], "hand": [[2, 2], [3, 2]]},
                {"bank": [10], "bet": [0], "hand": []},
            ],
        }
    )
    assert len(s.players[0].hand) == 2
    assert s.current_player == 0

    # Act - make a hit on the action
    skip_action = ActionInstance(acn.ActionName.SKIP, True)
    returned_state, decision, next_state, _ = gm.decide_hit_miss(s, action=skip_action)

    # Assert, ensure that we get another card, and same action
    assert len(s.players[0].hand) == 2, "Player one get another card"
    # next player
    assert s.current_player == 1
    action_space_possible = decision.action_range_to_numpy()
    assert action_space_possible == {
        acn.ActionName.SKIP.name: [0],
        acn.ActionName.HIT.name: [0],
        acn.ActionName.BET.name: [1, 10],
    }


def test_evaluate_round_end():
    s = State.from_data(
        {
            "current_player": 1,
            "hit_rounds": 0,
            "deck": [
                [8, 2],
                [6, 1],
                [11, 3],
                # ... snipped extra cards
            ],
            "discarded": [],
            "number_of_rounds": 0,
            "players": [
                # 5 points
                {"bank": [7], "bet": [3], "hand": [[2, 2], [3, 2]]},
                # 6 points - winner - should gets 5 gold
                {"bank": [10], "bet": [2], "hand": [[2, 2], [4, 2]]},
            ],
        }
    )
    assert s.current_player == 1
    assert s.players[1].bank == [10]

    # Act - make a hit on the action
    skip_action = ActionInstance(acn.ActionName.SKIP, True)
    returned_state, decision, next_state, _ = gm.decide_hit_miss(s, action=skip_action)

    # Assert, ensure that we get another card, and same action
    assert s.current_player == 0
    assert s.players[0].bet == [0]
    assert s.players[1].bet == [0]
    assert s.players[1].bank == [15]

    assert len(s.discarded) == 4, "4 cards discarded"
    # Go back to player 0 and place bet.
    assert next_state == gm.GameState.place_bet
    action_space_possible = decision.action_range_to_numpy()
    assert action_space_possible == {
        acn.ActionName.SKIP.name: [0],
        acn.ActionName.HIT.name: [0],
        # Note now player 1 have less money
        acn.ActionName.BET.name: [1, 7],
    }


def test_find_winner():
    s = State.from_data(
        {
            "current_player": 1,
            "hit_rounds": 0,
            # Last round
            "number_of_rounds": 3,
            "deck": [
                [8, 2],
                [6, 1],
                [11, 3],
                # ... snipped extra cards
            ],
            "discarded": [],
            "players": [
                # 5 points
                {"bank": [7], "bet": [3], "hand": [[2, 2], [3, 2]]},
                # 6 points - winner - should gets 5 gold
                {"bank": [10], "bet": [2], "hand": [[2, 2], [4, 2]]},
            ],
        }
    )
    assert s.current_player == 1

    # Act - make a hit on the action
    skip_action = ActionInstance(acn.ActionName.SKIP, True)
    returned_state, decision, next_state, _ = gm.decide_hit_miss(s, action=skip_action)

    # Now player 1 win!
    assert decision is None
    assert next_state == gm.GameState.end
