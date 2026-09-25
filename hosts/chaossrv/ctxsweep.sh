#!/usr/bin/env bash
# Context sweep: cold (salted --unique filler) prefill + 1024-token forced greedy decode at depth, c1 per depth, c4 at 32k/128k.
OUT=$HOME/appdata/flashnext/bench/${1:-ctxsweep}; mkdir -p $OUT
M=Qwen3.8-Flash-Next-EXL3-2.50bpw; P=$HOME/appdata/flashnext-2x5090/bench/probe.py; U=http://127.0.0.1:8022/v1
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,clocks.sm,memory.used --format=csv,noheader,nounits -lms 500 > $OUT/power.csv &
PW=$!; salt=$(( $(date +%s) % 100000 ))
for kind in code prose; do
  for ctx in 0 8000 16000 32000 64000 128000 200000 240000; do
    salt=$((salt+97))
    python3 $P --url $U --model $M --tag $kind-c1-$ctx --conc 1 --tokens 1024 --runs 1 --kind $kind --ctx $ctx --unique --salt $salt --distinct --out $OUT/$kind-c1.jsonl >> $OUT/$kind-c1.log 2>&1
  done
done
for ctx in 32000 128000; do salt=$((salt+97))
  python3 $P --url $U --model $M --tag code-c4-$ctx --conc 4 --tokens 1024 --runs 1 --kind code --ctx $ctx --unique --salt $salt --distinct --out $OUT/code-c4.jsonl >> $OUT/code-c4.log 2>&1
done
kill $PW; date > $OUT/done
