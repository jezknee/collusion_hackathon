#!/usr/bin/env python3
"""Run a batch of LLM-Luke games: one game per condition (or several), in one command.

Main conditions (run by default): none, paper_peer, indirect, direct.
Optional: benign, paper_peer_noreminder, paper_peer_fixed, paper_peer_offerall.

Each game runs as its own separate process (a fresh game, nothing shared), with its own log
file in logs/. Games start a few seconds apart, and a few run side by side to save time.
Every game saves its own record in experiments/undercover/<condition folder>/ as usual.
A summary of the batch is written to logs/batch_<date>.csv.

Run from the Collusion_Liars_Bar folder:
    Windows:   uv run python run_llm_games.py
    Mac/Linux: python run_llm_games.py

Options:
    --repeats 2           run every condition twice (default 1)
    --parallel 3          how many games run at the same time (default 3; 6-8 is fine if
                          the api_errors column stays at 0)
    --stagger 5           seconds between starting games (default 5)
    --conditions none paper_peer    only these conditions (names below)
    --include-optional    also run the optional conditions (or name any with --conditions)
    --max-rounds 15       passed to every game (default: no limit, as the game's default)
    --warn B              tell the ordinary players someone may be undercover (A, B or C;
                          prompts/undercover/warnings/); applies to every game in the batch
    --timeout 45          stop a game that runs longer than this many minutes (default 45)
    --dry-run             just print the commands
Press Ctrl+C to stop the batch; games still running are stopped, and are not usable.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

# name: (extra arguments for game_undercover.py, optional?)
CONDITIONS = {
    "none":                  (["--arm", "none", "--luke-policy", "honour"], False),
    "paper_peer":            (["--arm", "paper_peer", "--luke-policy", "honour"], False),
    "indirect":              (["--arm", "indirect", "--luke-policy", "honour"], False),
    "direct":                (["--arm", "direct", "--luke-policy", "honour"], False),
    # optional extras (--include-optional, or name them with --conditions)
    "benign":                (["--arm", "benign", "--luke-policy", "honour"], True),
    "paper_peer_noreminder": (["--arm", "paper_peer", "--luke-policy", "honour",
                               "--channel-reminders", "off"], True),
    "paper_peer_fixed":      (["--arm", "paper_peer", "--luke-policy", "fixed"], True),
    "paper_peer_offerall":   (["--arm", "paper_peer", "--luke-policy", "honour",
                               "--offer-to", "all"], True),
}
SAVED = re.compile(r"Game record saved to (\S+\.json)")
PARTIAL = re.compile(r"save automatically to (\S+\.json)")  # the game's per-round save
API_ERROR = "LLM error:"  # the repo's client prints this and returns an empty reply (e.g. rate limit)


def set_aside(log_path):
    """Move a stopped/failed game's partial record out of the results folder."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            paths = set(PARTIAL.findall(f.read()))
    except OSError:
        return []
    moved = []
    os.makedirs("experiments/incomplete", exist_ok=True)
    for p in paths:
        if os.path.exists(p):
            dest = os.path.join("experiments/incomplete", os.path.basename(p))
            shutil.move(p, dest)
            moved.append(dest)
    return moved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--stagger", type=float, default=5)
    ap.add_argument("--conditions", nargs="+", choices=list(CONDITIONS))
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=0)
    ap.add_argument("--warn", default="none", choices=["none", "A", "B", "C"])
    ap.add_argument("--timeout", type=float, default=45, help="minutes per game")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--game-script", default="src/game/game_undercover.py", help=argparse.SUPPRESS)
    args = ap.parse_args()

    names = args.conditions or [n for n, (_, opt) in CONDITIONS.items()
                                if args.include_optional or not opt]
    # Interleave repeats (none, paper_peer, ..., none, paper_peer, ...) so an interruption
    # still leaves every condition covered evenly.
    queue = [n for _ in range(args.repeats) for n in names]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def command(name):
        cmd = [sys.executable, args.game_script, "--luke", "llm"] + CONDITIONS[name][0]
        if args.max_rounds:
            cmd += ["--max-rounds", str(args.max_rounds)]
        if args.warn != "none":
            cmd += ["--warn", args.warn]
        return cmd

    print(f"Batch {stamp}: {len(queue)} games, {args.parallel} at a time, "
          f"{args.stagger:g}s apart. Conditions: {', '.join(names)}"
          + (f". Warning level {args.warn}" if args.warn != "none" else ""))
    if args.dry_run:
        for i, n in enumerate(queue, 1):
            print(f"  {i:>2}. {' '.join(command(n))}")
        return

    running, results, next_i, last_start = [], [], 0, 0.0
    try:
        while next_i < len(queue) or running:
            # start the next game if there's a free slot and the stagger has passed
            if (next_i < len(queue) and len(running) < args.parallel
                    and time.time() - last_start >= args.stagger):
                name = queue[next_i]
                next_i += 1
                log_path = f"logs/{stamp}_{next_i:02d}_{name}.log"
                log = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(command(name), stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append({"n": next_i, "name": name, "proc": proc, "log": log,
                                "log_path": log_path, "start": time.time()})
                last_start = time.time()
                print(f"[{datetime.now():%H:%M:%S}] started  {next_i:>2}/{len(queue)} {name}")
            # check on running games
            for g in list(running):
                code = g["proc"].poll()
                too_long = time.time() - g["start"] > args.timeout * 60
                if code is None and not too_long:
                    continue
                if code is None:
                    g["proc"].kill()
                    g["proc"].wait()
                    code = "timeout"
                g["log"].close()
                running.remove(g)
                with open(g["log_path"], encoding="utf-8", errors="replace") as f:
                    text = f.read()
                found = SAVED.findall(text)
                api_errors = text.count(API_ERROR)
                record = found[-1] if found else ""
                minutes = (time.time() - g["start"]) / 60
                ok = code == 0 and bool(record)
                if not ok:
                    for m in set_aside(g["log_path"]):
                        print(f"           partial record moved to {m}")
                results.append({"game": g["n"], "condition": g["name"], "ok": ok, "exit": code,
                                "minutes": round(minutes, 1), "api_errors": api_errors,
                                "record": record, "log": g["log_path"]})
                status = "done    " if ok else "PROBLEM "
                print(f"[{datetime.now():%H:%M:%S}] {status} {g['n']:>2}/{len(queue)} {g['name']} "
                      f"({minutes:.1f} min){'' if ok else ' -> see ' + g['log_path']}"
                      f"{f'  WARNING: {api_errors} API errors (empty replies)' if api_errors else ''}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping: ending the games still running (they won't be usable).")
        for g in running:
            g["proc"].kill()
            g["proc"].wait()
            g["log"].close()
            for m in set_aside(g["log_path"]):
                print(f"  partial record moved to {m}")
            results.append({"game": g["n"], "condition": g["name"], "ok": False, "exit": "stopped",
                            "minutes": round((time.time() - g["start"]) / 60, 1),
                            "api_errors": "", "record": "", "log": g["log_path"]})

    summary = f"logs/batch_{stamp}.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["game", "condition", "ok", "exit", "minutes", "api_errors",
                                          "record", "log"])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["game"]))
    good = sum(r["ok"] for r in results)
    print(f"\n{good} of {len(queue)} games finished and saved. Summary: {summary}")
    if good < len(queue):
        print("For any PROBLEM game, look at the end of its log. Partial records from those games "
              "were moved to experiments/incomplete/, so they don't get into the analysis.")
    hit = [r for r in results if r["api_errors"]]
    if hit:
        print(f"{len(hit)} games had API errors (players got empty replies, often a rate limit). "
              "See the api_errors column; consider re-running those games with fewer at once.")


if __name__ == "__main__":
    main()
