#!/usr/bin/env bash
# Host tuning used for the chaossrv numbers (all runtime-only; a reboot restores defaults). Needs sudo and nvidia-ml-py.
# PY = a python with nvidia-ml-py (pynvml) installed.
set -eu
PY=${PY:-python3}
for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo performance | sudo tee "$f" >/dev/null; done
for s in /sys/devices/system/cpu/cpu*/cpuidle/state[2-9]/disable; do [ -e "$s" ] && echo 1 | sudo tee "$s" >/dev/null; done   # no C2/C3
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -pl "${PL:-600}" >/dev/null
sudo "$PY" -c "import pynvml as N; N.nvmlInit(); hs=[N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]; [N.nvmlDeviceSetMemClkVfOffset(h, ${MEMOC:-4500}) for h in hs]; print('memoc', [N.nvmlDeviceGetMemClkVfOffset(h) for h in hs])"
nvidia-smi --query-gpu=index,power.limit,clocks.max.mem --format=csv,noheader
