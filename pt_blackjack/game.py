from typing import List, Tuple, Generator, Optional, Sequence, Type
from dataclasses import dataclass
import re
import numpy as np

import gym.spaces as spaces

from playtest.action import ActionInstance, ActionRange

from playtest.game import Game, Player
from pt_blackjack.state import State, PlayerState
from pt_blackjack.constant import Param, Reward
from pt_blackjack.action import (
    ActionFactory,
    ActionBetRange,
    ACTION_HIT,
    ACTION_SKIP,
    ActionBet,
    ActionHitRange,
    ActionSkipRange,
)


class Blackjack(Game[State, ActionFactory, Param]):

    base_state = State

    def __init__(self, param: Param, **kwargs):
        self.state = State(param=param)
        self.action_factory = ActionFactory(param=param)
        self.last_player_reward = Reward.VALID_ACTION

        super().__init__(param, **kwargs)

        self.players = []
        for i in range(param.number_of_players):
            self.players.append(Player(i))

    def start(
        self,
    ) -> Generator[
        # return: player_id, possible action, last_player_reward
        Tuple[int, Sequence[ActionRange], int],
        ActionInstance,  # receive: action
        None,  # terminal
    ]:
        winner = None
        for r in range(self.param.number_of_rounds):
            self.announcer.say("Start of next round!")
            for p in self.players:
                yield from self.deal_round(p)

            self.determine_winner()
            self.reset_hand()

            winner = self.find_winner()
            if winner is not None:
                break
        # TODO: how to reward the player?

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

    def deal_round(self, p: Player):
        self.a.say("Player {} - it is your turn!".format(p.id))

        player_state: PlayerState = self.s.get_player_state(p.id)
        self.a.say("Let's see your two card.")
        self.s.deck.deal(player_state.hand, 2)

        self.a.ask("How much you want to bet?")
        accepted_action: Sequence[ActionRange] = [ActionBetRange(self.state, p.id)]
        bet_value = None
        bet = yield from self.get_player_action(p.id, accepted_action)
        assert isinstance(bet, ActionBet)
        assert accepted_action[0].is_valid(
            bet
        ), f"Invalid bet made: {bet} based on {accepted_action[0]}"
        bet_value = bet.value
        self.a.say(f"Player {p.id} bet: {bet_value} coin")
        player_state.bet.take_from(player_state.bank, value=bet_value)
        action = None
        # TODO: calculate reward
        self.set_last_player_reward(Reward.BETTED)

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
