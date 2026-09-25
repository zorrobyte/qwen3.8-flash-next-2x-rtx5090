#!/usr/bin/env bash
# decode curve per adrienbrault R719: greedy, 1024 forced tokens, --distinct, warmup 1 + 3 runs, c1..c8, code+prose; GPU watts logged
TAG=${1:?tag}; OUT=$HOME/appdata/flashnext/bench/$TAG; mkdir -p $OUT
M=${MODEL:-Qwen3.8-Flash-Next-EXL3-2.50bpw}
[ "${POWER:-1}" = 1 ] && nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,clocks.sm,clocks.mem,memory.used --format=csv,noheader,nounits -lms 250 > $OUT/power.csv &
PW=$!; [ "${POWER:-1}" = 1 ] || PW=
for kind in code prose; do
  python3 $HOME/appdata/flashnext-2x5090/bench/probe.py --url http://127.0.0.1:8022/v1 --model $M --tag $TAG-$kind \
    --conc ${CONCS:-1 2 3 4 5 6 7 8} --tokens 1024 --runs ${RUNS:-3} --warmup-runs 1 --kind $kind --distinct \
    --out $OUT/$kind.jsonl > $OUT/$kind.log 2>&1
  date +%s > $OUT/$kind.done
done
[ -n "$PW" ] && kill $PW
