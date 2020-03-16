import pytest

from .action import (
    ActionBetRange,
    ActionHitRange,
    ActionSkipRange,
    ActionBet,
    ACTION_HIT,
    ACTION_SKIP,
)

from .constant import Param
from .state import State, PlayerState
from .game import Blackjack


@pytest.fixture
def game() -> Blackjack:
    param = Param(number_of_players=2, number_of_rounds=1)
    return Blackjack(param=param)


def test_start(game: Blackjack):
    game_gen = game.start()
    s = game.s

    assert len(s.players[0].hand) == 0
    assert len(s.players[1].hand) == 0

    # ability to forward the generator
    player_id, possible_actions, _ = next(game_gen)
    assert player_id == game.players[0].id
    assert possible_actions == [ActionBetRange(s, player_id)]


def test_round(game: Blackjack):
    s: State = game.state
    announcer = game.get_announcer()

    game_gen = game.deal_round(game.players[0])
    player_id, possible_actions, _ = next(game_gen)

    assert player_id == game.players[0].id
    assert possible_actions == [ActionBetRange(s, player_id)]
    assert (
        len(s.get_player_state(player_id).hand) == 2
    ), "Player should start with two cards"

    player_id, possible_actions, _ = game_gen.send(ActionBet(2))
    assert player_id == 0
    assert possible_actions == [
        ActionHitRange(s, player_id),
        ActionSkipRange(s, player_id),
    ]
    player_state: PlayerState = s.get_player_state(player_id)
    assert player_state.bet.amount == 2
    assert player_state.bank.amount == 8
    # TODO: implement announcer
    # assert any(m.contains("hit or pass") for m in announcer.messages)

    announcer.clear()
    player_id, possible_actions, _ = game_gen.send(ACTION_HIT)
    assert player_id == 0

    # TODO: implement announcer
    # assert any(m.contains("hit or pass") for m in announcer.messages)
    assert (
        len(s.get_player_state(player_id).hand) == 3
    ), "Player should receive extra card"

    announcer.clear()
    with pytest.raises(StopIteration):
        game_gen.send(ACTION_SKIP)
        # player, possible_actions = next(game_gen)
        # assert player == s.players[1], "Should forward to next player"

    # reset hand should reset all player's hand
    game.reset_hand()
    for p in game.players:
        assert len(s.get_player_state(p.id).hand) == 0
    assert len(s.discarded) > 0


def test_find_winner(game: Blackjack):
    s: State = game.state
    winner = game.find_winner()
    assert winner is None

    # Make player 0 the loser!
    s.get_player_state(0).bank.amount = 0

    # Now player 1 win!
    winner = game.find_winner()
    assert winner == 1
