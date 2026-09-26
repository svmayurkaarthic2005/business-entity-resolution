import subprocess
import time

r = subprocess.run("date +%H:%M; ps aux | grep 'src/exp_union' | grep -v grep | awk '{print $3\"%cpu \"$4\"%mem \"$10}'; free -g | sed -n 2p; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
                   "grep -vE 'Warning|warn' /content/exp_union.log | cut -c1-170 | tail -n 22",
                   shell=True, capture_output=True, text=True)
print(r.stdout + r.stderr)
