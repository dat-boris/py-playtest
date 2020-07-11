from typing import List, Tuple, Generator, Optional, Sequence, Type
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


def start(s: State, action=None) -> Tuple[State, acn.ActionDecision, GameState]:
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


def deal_round(s: State, action=None) -> Tuple[State, acn.ActionDecision, GameState]:
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
        GameState.decide_hit_pass,
    )


def handle_bet(s: State, action: ActionInstance):
    assert action.key == acn.ActionName.BET
    bet_value = action.value

    current_player = s.current_player
    player_state: PlayerState = s.get_player_state(current_player)

    logging.info(f"Player {current_player} bet: {bet_value} coin")
    player_state.bet.take_from(player_state.bank, value=bet_value)

    return (
        s,
        acn.ActionDecision({acn.ActionName.HIT: True, acn.ActionName.SKIP: True,}),
        GameState.decide_hit_pass,
    )


def decide_hit_miss(s: State, action: ActionInstance):
    hit_rounds = 0
    self.a.ask("Do you want to hit or pass?")
    while action != ACTION_SKIP and hit_rounds <= self.param.max_hits:
        # Note that this is class of action, while returned would be an instance
        accepted_action = [
            ActionHitRange(self.state, p.id),
            ActionSkipRange(self.state, p.id),
        ]
        action = yield from self.get_player_action(p.id, accepted_action)
        self.a.say(f"Player {p.id} round {hit_rounds}: {action}")
        assert any(
            [action_range.is_valid(action) for action_range in accepted_action]
        ), f"Invalid action: {action}"
        if action == ACTION_HIT:
            self.s.deck.deal(player_state.hand, 1)
            # TODO: calculate reward
            self.set_last_player_reward(Reward.HITTED)

        hit_rounds += 1

    # TODO: calculate reward
    self.set_last_player_reward(Reward.SKIPPED)
    self.a.say("Okay pass - on to next player!")


def find_winner(self) -> Optional[int]:
    all_banks = [self.s.get_player_state(p.id).bank.amount for p in self.players]
    has_broke = any([b <= 1 for b in all_banks])
    if not has_broke:
        return None

    winner = None
    winner_amount = -0xFFFF
    for i, v in enumerate(all_banks):
        if v > winner_amount:
            winner = i
    return winner


def reset_hand(self):
    for p in self.players:
        player_state = self.s.get_player_state(p.id)
        player_state.hand.deal(self.state.discarded, all=True)


def determine_winner(self):
    all_score = {}
    losers = []
    winner = None
    for p in self.players:
        ps = self.s.get_player_state(p.id)
        score = sum([c.number for c in ps.hand])
        if score > self.param.max_score:
            self.a.say("Player {} is busted! ({})".format(p.id, score))
            losers.append(p)
        else:
            self.a.say("Player {} has {} points!".format(p.id, score))
            all_score[p] = score

    if not all_score:
        self.a.say("All busted, no winner!")
        return
    else:
        sorted_players = sorted(
            all_score.keys(), key=lambda v: all_score[v], reverse=True
        )
        # TODO: what happen if equal score?
        winner = sorted_players[0]
        self.a.say("Player {} is the winner!".format(winner.id))
        losers.extend(sorted_players[1:])

    winner_pot = self.s.get_player_state(winner.id).bank
    total_bets = 0
    for p in self.players:
        total_bets += self.s.get_player_state(p.id).bet.amount
        winner_pot.take_from(self.s.get_player_state(p.id).bet)
    self.a.say("Player {} gains {} gold!".format(p.id, total_bets))
