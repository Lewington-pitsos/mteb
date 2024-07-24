#!/bin/bash

LOGFILE="gpu_utilization.log"

while true; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "Timestamp: %s, GPU Utilization: %3s%%, Memory Utilization: %6.2f%%\n", strftime("%Y-%m-%d %H:%M:%S"), $1, ($2/$3)*100}' >> $LOGFILE
  sleep 1
done
