from pprint import pprint
import pytest

from .constant import Param
from .state import State, PlayerState
import pt_blackjack.game as gm
import pt_blackjack.action as acn


NUMBER_OF_PLAYERS = 2


def test_start():
    s = State(Param(number_of_players=NUMBER_OF_PLAYERS))
    assert len(s.players[0].hand) == 0, "First hand is empty"

    returned_state, decision, next_state = gm.start(s)

    assert next_state == gm.GameState.decide_hit_pass
    # TODO: ensure we can do comparison
    # assert s == returned_state

    assert len(s.players[0].hand) == 2, "First hand is dealed"
    assert len(s.players[1].hand) == 0, "Second hand is not dealed"
    pprint(s.to_data())


@pytest.mark.xfail
def test_to_next_bet():
    # ability to forward the generator
    player_id, possible_actions, _ = next(game_gen)
    assert player_id == game.players[0].id
    assert possible_actions == [ActionBetRange(s, player_id)]


@pytest.mark.xfail
def test_round():
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


@pytest.mark.xfail
def test_find_winner():
    s: State = game.state
    winner = game.find_winner()
    assert winner is None

    # Make player 0 the loser!
    s.get_player_state(0).bank.value = [0]

    # Now player 1 win!
    winner = game.find_winner()
    assert winner == 1
