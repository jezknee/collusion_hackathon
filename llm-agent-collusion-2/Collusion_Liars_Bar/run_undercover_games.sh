#!/usr/bin/env bash
# Run several undercover games at the same time, each in the background, with a log file each.
# Usage (from the Collusion_Liars_Bar folder):
#   ./run_undercover_games.sh direct 4      # four games of the "direct" arm
# Arms: none, direct, indirect, benign
set -u
ARM="$1"; COUNT="${2:-4}"
mkdir -p logs
for i in $(seq 1 "$COUNT"); do
  python src/game/game_undercover.py --arm "$ARM" > "logs/${ARM}_$(date +%H%M%S)_$i.log" 2>&1 &
  echo "Started $ARM game $i (log in logs/)"
  sleep 3   # stagger the starts slightly
done
echo "All started. Check progress with:  tail -f logs/*.log    Wait for all with:  wait"
wait
echo "All $COUNT $ARM games finished. Results are in experiments/undercover/$ARM/"
