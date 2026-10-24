#!/usr/bin/env bash
# Run several undercover games at the same time, each in the background, with a log file each.
# Usage (from the Collusion_Liars_Bar folder):
#   ./run_undercover_games.sh direct 4 honour llm   # four LLM-Luke "direct" games, honour policy
#   ./run_undercover_games.sh direct 4 honour llm --channel-reminders off   # extra options at the end
# Arms: none, paper_peer, direct, indirect, benign.  Policies: fixed, never, honour.
# Luke: scripted (default) or llm.
set -u
ARM="$1"; COUNT="${2:-4}"; POLICY="${3:-fixed}"; LUKE="${4:-scripted}"; shift 4 2>/dev/null || shift $#
EXTRA="$*"
mkdir -p logs
for i in $(seq 1 "$COUNT"); do
  python src/game/game_undercover.py --arm "$ARM" --luke-policy "$POLICY" --luke "$LUKE" $EXTRA > "logs/${ARM}-${POLICY}-${LUKE}_$(date +%H%M%S)_$i.log" 2>&1 &
  echo "Started $ARM-$POLICY ($LUKE Luke) game $i (log in logs/)"
  sleep 3   # stagger the starts slightly
done
echo "All started. Check progress with:  tail -f logs/*.log    Wait for all with:  wait"
wait
echo "All $COUNT games finished. Results are in experiments/undercover/"
