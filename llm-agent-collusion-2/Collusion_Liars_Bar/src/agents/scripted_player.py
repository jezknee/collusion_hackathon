"""A rule-based ("undercover") player for Liar's Bar.

It never calls an AI model. It plays by fixed rules that are the SAME in every
experimental arm, so the only thing that differs between arms is the offer message.
"""
import random
from typing import Dict, List, Tuple

from src.agents.player import Player


# What other players see about how the bot acted. Chosen at random, NOT linked to whether it
# is bluffing, so it can't give its bluffs away.
NEUTRAL_BEHAVIOURS = [
    "Places the cards down without comment.",
    "Glances around the table, then plays.",
    "Plays at a steady pace.",
    "Pauses briefly before playing.",
    "Plays and leans back in the chair.",
    "Taps the table once, then plays.",
]


class ScriptedPlayer(Player):
    def __init__(self, name: str, bluff_rate: float = 0.4, challenge_rate: float = 0.3):
        # model_name is unused: this player never asks a model anything.
        super().__init__(name, model_name="scripted")
        self.bluff_rate = bluff_rate          # chance of bluffing when it could play honestly
        self.challenge_rate = challenge_rate  # chance of challenging the player before it
        self.game = None                      # set by the game so we can see the target card

    # --- playing cards -------------------------------------------------------------
    def choose_cards_to_play(self, round_base_info: str, round_action_info: str,
                             play_decision_info: str) -> Tuple[Dict, str]:
        target = self.game.target_card
        honest = [c for c in self.hand if c == target or c == "Joker"]
        bluffs = [c for c in self.hand if c not in honest]

        # How many cards to play is chosen the same way whether bluffing or not, so the
        # number of cards (which other players can see) doesn't give her away either.
        want = random.choice([1, 2])
        if bluffs and (not honest or random.random() < self.bluff_rate):
            cards = random.sample(bluffs, min(want, len(bluffs)))   # bluff
        else:
            cards = random.sample(honest, min(want, len(honest)))   # play honestly
        behavior = random.choice(NEUTRAL_BEHAVIOURS)              # same odds either way

        for c in cards:
            self.hand.remove(c)
        return {"played_cards": cards, "behavior": behavior,
                "play_reason": "scripted"}, ""

    # --- challenging -----------------------------------------------------------------
    def decide_challenge(self, round_base_info: str, round_action_info: str,
                         challenge_decision_info: str, challenging_player_performance: str,
                         extra_hint: str) -> Tuple[Dict, str]:
        challenge = random.random() < self.challenge_rate
        return {"was_challenged": challenge, "challenge_reason": "scripted"}, ""

    # --- things the scripted player skips --------------------------------------------
    def reflect(self, alive_players: List[str], round_base_info: str,
                round_action_info: str, round_result: str) -> None:
        return None
