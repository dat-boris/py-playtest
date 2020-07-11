from typing import List, Tuple, Generator, Optional, Sequence, Type, Dict
from dataclasses import dataclass
import logging
import enum
import re
import numpy as np

import gym.spaces as spaces

from playtest.action import ActionInstance, ActionRange
from playtest.game import Game, Player

import pt_blackjack.action as acn
from pt_blackjack.state import State, PlayerState
from pt_blackjack.constant import Param, Reward


class GameState(enum.Enum):
    """## Game round

    At each round, the play goes in turn to make the decision.

    1. Place worker as necessary in clockwise
    2. Resolve actions of each place - you may play special ability
       of cards that you acquire.

    Then it goes to the next player.
    """

    start = "start"
    place_bet = "place_bet"
    decide_hit_pass = "decide_hit_pass"
    end_of_round = "end_of_round"
    end = "end"


TypeReturnState = Tuple[State, Optional[acn.ActionDecision], GameState]


def start(s: State, action=None) -> TypeReturnState:
    logging.info("Start of next round!")
    current_player = s.current_player
    return deal_round(s)

    # TODO: oh well, this is what happen! We done have a loop!
    # self.determine_winner()
    # self.reset_hand()

    # winner = self.find_winner()
    # if winner is not None:
    #     break
    # TODO: how to reward the player?


def deal_round(s: State, action=None) -> TypeReturnState:
    current_player = s.current_player
    logging.info(f"Player {current_player} - it is your turn!")

    player_state: PlayerState = s.get_player_state(current_player)
    logging.info("Let's see your two card.")
    s.deck.deal(player_state.hand, 2)

    logging.info("How much you want to bet?")
    bank_value = player_state.bank.value[0]

    return (
        s,
        acn.ActionDecision({acn.ActionName.BET: (Param.min_bet_per_round, bank_value)}),
        GameState.place_bet,
    )


def handle_bet(s: State, action: ActionInstance) -> TypeReturnState:
    assert action.key == acn.ActionName.BET
    bet_value = action.value

    current_player = s.current_player
    player_state: PlayerState = s.get_player_state(current_player)

    logging.info(f"Player {current_player} bet: {bet_value} coin")
    player_state.bet.take_from(player_state.bank, value=bet_value)

    logging.info("Do you want to hit or pass?")
    return (
        s,
        # TODO: this action is really tied to the state (instead of others)
        acn.ActionDecision({acn.ActionName.HIT: True, acn.ActionName.SKIP: True,}),
        GameState.decide_hit_pass,
    )


def decide_hit_miss(s: State, action: ActionInstance) -> TypeReturnState:
    assert action.key in {acn.ActionName.HIT, acn.ActionName.SKIP}

    current_player = s.current_player
    player_state: PlayerState = s.get_player_state(current_player)
    hit_rounds = s.hit_rounds

    logging.info(f"Player {current_player} round {hit_rounds}: {action}")

    if action.key == acn.ActionName.HIT:
        s.deck.deal(player_state.hand, 1)
        # TODO: calculate reward
        # self.set_last_player_reward(Reward.HITTED)
        s.hit_rounds += 1
        return (
            s,
            acn.ActionDecision({acn.ActionName.HIT: True, acn.ActionName.SKIP: True,}),
            GameState.decide_hit_pass,
        )
    elif action.key == acn.ActionName.SKIP:
        logging.info("Okay pass - on to next player!")
        # TODO: calculate reward
        # self.set_last_player_reward(Reward.SKIPPED)
        s.hit_rounds = 0
        s.next_player()
        next_player_bank = s.players[s.current_player].bank

        # do we have more players to play?
        if s.current_player == 0:
            # End of round - check for winner
            return check_winner(s)

        # Next player bet
        # TODO: really should be just going to state?
        return (
            s,
            acn.ActionDecision(
                {
                    acn.ActionName.BET: (
                        Param.min_bet_per_round,
                        next_player_bank.value[0],
                    )
                }
            ),
            GameState.decide_hit_pass,
        )

    raise RuntimeError(f"Unknown action {action}")


def check_winner(s: State, action=None) -> TypeReturnState:
    current_player = s.current_player

    # Now check for winner
    all_score: Dict[int, int] = {}
    losers: List[int] = []
    winner: Optional[int] = None
    p: PlayerState
    for player_id, p in enumerate(s.players):
        ps = s.get_player_state(player_id)
        score_in_hand = sum([c.number for c in ps.hand])
        if score_in_hand > Param.max_score:
            logging.info("Player {} is busted! ({})".format(player_id, score_in_hand))
            # losers.append(p)
        else:
            logging.info("Player {} has {} points!".format(player_id, score_in_hand))
            all_score[player_id] = score_in_hand

    if not all_score:
        logging.info("All busted, no winner!")
        # No winner!
        # TODO: recording the winner
    else:
        sorted_players: List[int] = sorted(
            all_score.keys(), key=lambda v: all_score[v], reverse=True
        )

        # TODO: what happen if equal score?
        winner = sorted_players[0]
        logging.info("Player {} is the winner!".format(winner))
        losers.extend(sorted_players[1:])

        winner_pot = s.get_player_state(winner).bank
        total_bets = 0
        for player_id, p in enumerate(s.players):
            total_bets += s.get_player_state(player_id).bet.amount
            winner_pot.take_from(s.get_player_state(player_id).bet)
        logging.info("Player {} gains {} gold!".format(player_id, total_bets))

    return end_of_round_next_round_check(s)


def end_of_round_next_round_check(s: State, action=None) -> TypeReturnState:
    current_player = s.current_player

    all_banks = [
        s.get_player_state(player_id).bank.amount
        for player_id, p in enumerate(s.players)
    ]
    has_broke = any([b <= 1 for b in all_banks])
    if has_broke:
        return find_final_winner(s)

    if current_player == 0:
        s.number_of_rounds += 1
        if s.number_of_rounds >= Param.number_of_rounds:
            logging.info("End of game - {s.number_of_rounds}")
            return find_final_winner(s)

    # Go back into betting
    return reset_round(s)


def reset_round(s: State, action=None) -> TypeReturnState:
    p: PlayerState
    for player_id, p in enumerate(s.players):
        # reset all hands
        p.hand.deal(s.discarded, all=True)
    s.hit_rounds = 0
    return deal_round(s)


def find_final_winner(s: State, action=None) -> TypeReturnState:
    all_banks = [s.get_player_state(p.id).bank.amount for p in s.players]
    winner = None
    winner_amount = -0xFFFF
    for i, v in enumerate(all_banks):
        if v > winner_amount:
            winner = i

    # TODO: setting up winner control
    return (s, None, GameState.end)

