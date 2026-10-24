#!/usr/bin/env python3
"""Offer acceptance test for the undercover experiment, built on the paper's own test.

This reuses the repo's SecretToolAcceptanceTest (src/test/test_secret_comm_acceptance.py)
and its peer-invitation test, `test_partner_response`, WITHOUT changing it: same game
context (the paper's four players), same message layout, same model settings, same reply
parsing. The only thing that changes between arms is the invitation text.

Arms:
    paper_peer  the paper's own peer invitation, prompts/secret_channel/partner_response_prompt.txt
                (a second positive control: an explicit alliance offered by a player)
    direct      prompts/undercover/direct.txt
    indirect    prompts/undercover/indirect.txt
    benign      prompts/undercover/benign.txt

The invitation always comes from Luke, the undercover player in our games; Lily, Mike
and Quinn receive it. The paper's designer offer (the main positive control) is run with
the paper's own script; see the guide, Part 4a.

Needs the one-line fix to the repo script first (line 45: 'tool_offer_prompt copy.txt'
-> 'tool_offer_prompt.txt'). Run from the Collusion_Liars_Bar folder:
    Mac/Linux:  PYTHONUTF8=1 python src/test/undercover_acceptance_test.py --n 20
    Windows:    $env:PYTHONUTF8="1"   then   uv run python src/test/undercover_acceptance_test.py --n 20
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)  # the repo script imports `llm.llm_client`

UNDERCOVER = "Luke"
RECEIVERS = ["Lily", "Mike", "Quinn"]
ARM_FILES = {
    "paper_peer": "prompts/secret_channel/partner_response_prompt.txt",
    "direct": "prompts/undercover/direct.txt",
    "indirect": "prompts/undercover/indirect.txt",
    "benign": "prompts/undercover/benign.txt",
}


def load_repo_test():
    """Import the repo's test module by file path (a module named 'test' clashes with Python's own)."""
    path = os.path.join(HERE, "test_secret_comm_acceptance.py")
    spec = importlib.util.spec_from_file_location("repo_acceptance_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SecretToolAcceptanceTest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="trials per arm per player")
    ap.add_argument("--arms", nargs="+", default=list(ARM_FILES), choices=list(ARM_FILES))
    ap.add_argument("--players", nargs="+", default=RECEIVERS, choices=RECEIVERS)
    args = ap.parse_args()

    Tester = load_repo_test()
    try:
        tester = Tester()  # loads the rules, prompts and API client exactly as the paper's test does
    except FileNotFoundError as e:
        sys.exit(f"\n{e}\nApply the one-line fix to src/test/test_secret_comm_acceptance.py first "
                 f"(line 45: 'tool_offer_prompt copy.txt' -> 'tool_offer_prompt.txt').")
    except UnicodeDecodeError:
        sys.exit("\nText-encoding error. Set PYTHONUTF8=1 first (Windows: $env:PYTHONUTF8=\"1\").")

    # Capture each raw reply without changing the repo's method.
    real_chat = tester.client.chat
    last = {}

    def recording_chat(messages, model, *a, **k):
        reply, reasoning = real_chat(messages, model, *a, **k)
        last["reply"] = reply
        return reply, reasoning

    tester.client.chat = recording_chat

    jobs = [(arm, p, i) for arm in args.arms for p in args.players for i in range(args.n)]
    random.shuffle(jobs)  # mix arms so a provider hiccup doesn't hit one arm only
    print(f"Running {len(jobs)} trials...")

    os.makedirs("test_results", exist_ok=True)
    out_path = f"test_results/undercover_acceptance_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    counts = defaultdict(Counter)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, (arm, player, i) in enumerate(jobs, 1):
            with open(ARM_FILES[arm], encoding="utf-8") as pf:
                tester.partner_response_prompt_template = pf.read()
            last.clear()
            accepted = tester.test_partner_response(player, tester.models[player], UNDERCOVER)
            reply = last.get("reply", "")
            decision = "ACCEPT" if accepted else "REFUSE"
            if not reply:
                decision = "ERROR"  # the repo counts these as refusals; we keep them separate
            row = {"arm": arm, "player": player, "model": tester.models[player], "trial": i,
                   "initiator": UNDERCOVER, "decision": decision, "reply": reply}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            counts[(arm, player)][decision] += 1
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)} done")

    print(f"\nSaved every reply to {out_path}\n")
    print(f"{'arm':<12}{'player':<8}{'accept':>8}{'refuse':>8}{'error':>7}{'accept %':>10}")
    for (arm, player), c in sorted(counts.items()):
        valid = c["ACCEPT"] + c["REFUSE"]
        pct = 100 * c["ACCEPT"] / valid if valid else 0
        print(f"{arm:<12}{player:<8}{c['ACCEPT']:>8}{c['REFUSE']:>8}{c['ERROR']:>7}{pct:>9.1f}%")


if __name__ == "__main__":
    main()
