#!/usr/bin/env bash
# VM-side watchdog (cron */5): one queue worker per GPU stays alive; an idle worker running stale worker.py is refreshed.
# Install once:  (crontab -l 2>/dev/null; echo "*/5 * * * * bash $HOME/openlatent/research/fields/scripts/watchdog.sh") | crontab -
W=$HOME/openlatent/research/fields/scripts/worker.py
LOG=$HOME/work/logs/watchdog.log
mkdir -p "$HOME/work/logs" "$HOME/queue/pending" "$HOME/queue/running"
code_mtime=$(stat -c %Y "$W")
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
for g in $(seq 0 $((ngpu - 1))); do
  pid=$(pgrep -f "worker.py --gpu $g\$" | head -1)
  if [ -z "$pid" ]; then
    setsid nohup python3 "$W" --gpu "$g" >> "$HOME/work/logs/worker$g.log" 2>&1 < /dev/null &
    echo "$(date -u +%FT%TZ) restarted missing worker $g" >> "$LOG"
    continue
  fi
  start=$(date -d "$(ps -o lstart= -p "$pid")" +%s 2>/dev/null || echo 0)
  busy=$(ls "$HOME/queue/running/gpu${g}__"* 2>/dev/null | wc -l)
  if [ "$code_mtime" -gt "$start" ] && [ "$busy" -eq 0 ]; then
    kill "$pid"; sleep 2
    setsid nohup python3 "$W" --gpu "$g" >> "$HOME/work/logs/worker$g.log" 2>&1 < /dev/null &
    echo "$(date -u +%FT%TZ) refreshed stale idle worker $g" >> "$LOG"
  fi
done

# express workers: reserved-priority (<= 4) short jobs never wait behind long runs (they share GPUs 0, 1, 2 with the regular workers)
for g in 0 1 2; do
  if ! pgrep -f "worker.py --gpu $g --max_priority 4" >/dev/null; then
    setsid nohup python3 "$W" --gpu "$g" --max_priority 4 >> "$HOME/work/logs/worker_express$g.log" 2>&1 < /dev/null &
    echo "$(date -u +%FT%TZ) started express worker on gpu $g" >> "$LOG"
  fi
done
