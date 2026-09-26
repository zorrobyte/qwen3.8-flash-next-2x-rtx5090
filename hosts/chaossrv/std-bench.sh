#!/usr/bin/env bash
# R731b's standard benchmark on chaossrv (rootless podman): `vllm bench serve` through bench/vllm_bench_tabby.py on
# ShareGPT V3 (400 prompts, seed 7310, reference-length outputs) and Spec-Bench (480 questions, 256 forced tokens) at
# 1/2/4/8 concurrent requests. Each cell boots the server afresh (empty prompt cache), warms it with `c` non-stream
# 64-token requests on prompts outside both datasets, runs the client closed loop, and counts foreign requests
# (server log entries without min_tokens) that arrived during the cell.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$HOME/appdata/flashnext/bench/std-${1:-run}}
DS=${DS:-$HOME/appdata/flashnext/datasets}
MDIR=${MDIR:-$HOME/appdata/models/Qwen3.8-Flash-Next-EXL3-2.50bpw}
MODEL=Qwen3.8-Flash-Next-EXL3-2.50bpw
CLIENT_IMG=${CLIENT_IMG:-docker.io/vllm/vllm-openai:v0.30.0}
CONCS=${CONCS:-"1 2 4 8"}; DATASETS=${DATASETS:-"sharegpt specbench"}
mkdir -p "$OUT/results" "$OUT/cells"
log(){ echo "$(date -Is) $*" | tee -a "$OUT/audit.log"; }
CENV=(-e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e VLLM_USE_RUST_BENCH=0 -e TABBY_FORCE_LEN=1)

warm(){ python3 - "$1" <<'EOF'
import json, sys, threading, urllib.request
c = int(sys.argv[1]); errs = []
def one(i):
    body = {"model": "Qwen3.8-Flash-Next-EXL3-2.50bpw", "stream": False, "max_tokens": 64, "min_tokens": 64, "temperature": 0,
            "messages": [{"role": "user", "content": f"Warm-up {i}: list the planets of the solar system in order."}]}
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8022/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"}), timeout=300).read()
    except Exception as e:
        errs.append(str(e)[:120])
ts = [threading.Thread(target=one, args=(i,)) for i in range(c)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"warm-up {c}x done, {len(errs)} errors {errs[:1]}")
EOF
}

foreign(){ python3 - "$1" <<'EOF'
import re, sys
t = open(sys.argv[1], errors="replace").read()
t = re.sub(r"\n\s{10,}", " ", t)
h = re.findall(r"INFO:\s+#(\d+) [\w/.-]+(?: \([\w-]+\))?: [\d,]+ prompt tokens ·\s+(.*)", t)
print(sum(1 for n, rest in h if "min_tokens" not in rest), len(h))
EOF
}

log "=== std-bench $CLIENT_IMG, datasets '$DATASETS', concs '$CONCS'"
for ds in $DATASETS; do
  for c in $CONCS; do
    tag=$ds-c$c
    systemctl --user restart flashnext || { log "[$tag] ABORT: boot failed"; exit 1; }
    since=$(date -Is)
    log "[$tag] booted; $(warm "$c" 2>&1 | tail -1)"
    case $ds in
      sharegpt)  n=400; dsargs=(--dataset-name sharegpt --dataset-path /data/ShareGPT_V3_unfiltered_cleaned_split.json --seed 7310) ;;
      specbench) n=480; dsargs=(--dataset-name spec_bench --dataset-path /data/spec_bench_question.jsonl --spec-bench-output-len 256 --seed 7310) ;;
    esac
    podman run --rm --network host --security-opt label=disable "${CENV[@]}" -e TABBY_SAMPLES_OUT="/out/$tag.samples.tsv" \
      -v "$MDIR":/model:ro -v "$DS":/data:ro -v "$REPO/bench":/probes:ro -v "$OUT/results":/out \
      --entrypoint python3 "$CLIENT_IMG" /probes/vllm_bench_tabby.py bench serve \
      --backend openai-chat --base-url http://127.0.0.1:8022 --endpoint /v1/chat/completions \
      --model "$MODEL" --tokenizer /model --temperature 0 \
      --request-rate inf --max-concurrency "$c" --num-warmups 0 --num-prompts "$n" --no-oversample \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --save-detailed --result-dir /out --result-filename "$tag.json" --disable-tqdm \
      --metadata "tag=$tag" "dataset=$ds" "conc=$c" "boot=fresh per cell" "host=chaossrv" "client=vllm-v0.30.0+vllm_bench_tabby" \
      "${dsargs[@]}" > "$OUT/cells/client-$tag.log" 2>&1
    rc=$?
    sleep 5; podman logs --since "$since" flashnext > "$OUT/cells/container-$tag.log" 2>&1
    read -r fr hdr < <(foreign "$OUT/cells/container-$tag.log")
    log "[$tag] rc $rc; $(grep -aE '^(Successful requests|Output token throughput)' "$OUT/cells/client-$tag.log" | tr -s ' ' | tr '\n' ' '); foreign $fr of $hdr requests"
  done
done
log "=== done"
