#!/usr/bin/env bash
# Reproduce tabbyapi:stack-r3-rows32 from this repo with rootless podman (no GPU needed). Fork fixes: see hosts/chaossrv/README.md.
# Usage: build-chain.sh [first-step-number]   -- resumes from that step.
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd)
O=$R/overlays
B=${BUILD_DIR:-$HOME/.cache/flashnext-build}
L=$B/logs; mkdir -p $L
J=${J:-32}                     # MAX_JOBS where the Dockerfile exposes it
MEM=${MEM:-72g}                # cap build containers so the host (and its benchmark) keeps headroom
START=${1:-0}
P='tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap-ppipe'

n=0
step() {  # step <tag> <podman build args...>
  local tag=$1; shift; n=$((n+1))
  if (( n < START )); then echo "skip $n $tag"; return; fi
  local log=$L/$(printf %02d $n)-${tag#tabbyapi:}.log
  echo "=== [$n] $tag  $(date -Is)" | tee -a $L/chain.log
  local t0=$SECONDS
  if podman build --memory "$MEM" --network host --format docker -t "$tag" "$@" >"$log" 2>&1; then
    echo "    ok $tag in $((SECONDS-t0)) s" | tee -a $L/chain.log
  else
    echo "    FAIL $tag after $((SECONDS-t0)) s, see $log" | tee -a $L/chain.log; exit 1
  fi
}

# staged context for moe-coop-v2 (its Dockerfile expects out/ with test_moe_coop_v2.py at the context root)
rm -rf $B/ctx/moecoopv2 && mkdir -p $B/ctx/moecoopv2
cp -a $O/moe-coop-v2-overlay $B/ctx/moecoopv2/moe-coop-v2-overlay
cp $O/moe-coop-v2-overlay/test_moe_coop_v2.py $B/ctx/moecoopv2/

step tabbyapi:53da7919-rqcount      -f $R/Dockerfile.tabbyapi $R
step tabbyapi:qsa-cid               -f $R/Dockerfile.tabbyapi-qsa-cid $R
step tabbyapi:qsa-cid-pr337         -f $R/Dockerfile.tabbyapi-pr337 --build-arg BASE=tabbyapi:qsa-cid $R
step tabbyapi:qsa-cid-pr337-bszn16  -f $R/Dockerfile.tabbyapi-bszn --build-arg BASE=tabbyapi:qsa-cid-pr337 --build-arg PATCH=bszn16.patch --build-arg MAX_JOBS=$J $R
step tabbyapi:qsa-cid-pr337-bszn16-coopwide -f $R/Dockerfile.tabbyapi-bszn --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16 --build-arg PATCH=coopwide.patch --build-arg MAX_JOBS=$J $R
step tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2 -f $R/Dockerfile.tabbyapi-hcmix2-refbase --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide --build-arg MAX_JOBS=$J $R
step tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap -f $R/Dockerfile.tabbyapi-pyfile --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2 --build-arg SRC=hostgap-gated_delta_net.py --build-arg DST=exllamav3/modules/gated_delta_net.py $R
step $P                             -f $R/Dockerfile.tabbyapi-pypatch --build-arg BASE=tabbyapi:qsa-cid-pr337-bszn16-coopwide-hcmix2-hostgap --build-arg PATCH=prefill-pipeline.patch $R
step $P-nosync                      -f $O/prefill-nosync-overlay/Dockerfile --build-arg BASE=$P $O/prefill-nosync-overlay
step $P-nosync-mtpfix2              -f $O/prefill-pipeline-mtp-overlay/Dockerfile --build-arg BASE=$P-nosync $O/prefill-pipeline-mtp-overlay
step $P-nosync-mtpfix2-moecoopv2    -f $B/ctx/moecoopv2/moe-coop-v2-overlay/Dockerfile --build-arg BASE=$P-nosync-mtpfix2 $B/ctx/moecoopv2
step tabbyapi:decode-kernels-r2     -f $O/decode-kernels-r2/Dockerfile.box --build-arg BASE=$P-nosync-mtpfix2-moecoopv2 $O/decode-kernels-r2
step tabbyapi:decode-kernels-r2-refbase -f $O/refbase/Dockerfile.box --build-arg BASE=tabbyapi:decode-kernels-r2 $O/refbase
step tabbyapi:decode-kernels-r4     -f $O/decode-kernels-r4/Dockerfile.box --build-arg BASE=tabbyapi:decode-kernels-r2-refbase $O/decode-kernels-r4
step tabbyapi:stack-r4-e3r2         -f $O/stack-r4-e3r2/Dockerfile.box --build-arg BASE=tabbyapi:decode-kernels-r4 $O/stack-r4-e3r2
step tabbyapi:mtp-pruned-r1         -f $O/mtp-pruned-r1/Dockerfile.box --build-arg BASE=tabbyapi:stack-r4-e3r2 $O/mtp-pruned-r1
step tabbyapi:mtp-pruned-r1-tc1     -f $O/tool-choice-r1/Dockerfile.box --build-arg BASE=tabbyapi:mtp-pruned-r1 $O/tool-choice-r1
step tabbyapi:mtp-pruned-r1-tc1-plefix -f $O/ple-ckpt-clone-r1/Dockerfile.box --build-arg BASE=tabbyapi:mtp-pruned-r1-tc1 $O/ple-ckpt-clone-r1
step tabbyapi:nvme-tier-r4          -f $O/nvme-tier-r4/Dockerfile.box --build-arg BASE=tabbyapi:mtp-pruned-r1-tc1-plefix $O/nvme-tier-r4
step tabbyapi:nvme-tier-r4-e3det    -f $O/e3-det-r1/Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4 $O/e3-det-r1
step tabbyapi:nvme-tier-r4-e3det-r6 -f $O/decode-kernels-r6/overlay/Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4-e3det $O/decode-kernels-r6
step tabbyapi:nvme-tier-r4-e3det-r6-rawk -f $O/qsa-rawk-ring-r1/Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4-e3det-r6 $O/qsa-rawk-ring-r1
step tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 -f $O/gdn-state-bf16-r1/Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk --build-arg MANIFEST=manifest-on-r6.json $O/gdn-state-bf16-r1
step tabbyapi:ngram-prefetch-r1-gdnbf16 -f $O/ngram-prefetch-r1/Dockerfile.box --build-arg BASE=tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16 $O/ngram-prefetch-r1
step tabbyapi:mtpwin-r2             -f $O/mtp-kv-window-r2/Dockerfile.box --build-arg BASE=tabbyapi:ngram-prefetch-r1-gdnbf16 $O/mtp-kv-window-r2
step tabbyapi:mtpwin-r2-metrics1    -f $O/metrics-r1/Dockerfile.box --build-arg BASE=tabbyapi:mtpwin-r2 $O/metrics-r1
step tabbyapi:bverify-r1            -f $R/Dockerfile.tabbyapi-bverify --build-arg BASE=tabbyapi:mtpwin-r2-metrics1 $R
step tabbyapi:mtpnorm-r1            -f $R/Dockerfile.tabbyapi-mtpnorm --build-arg BASE=tabbyapi:bverify-r1 $R
step tabbyapi:mixstate-r1 -f $R/Dockerfile.tabbyapi-mixstate --build-arg BASE=tabbyapi:mtpnorm-r1 --build-arg MAX_JOBS=$J $R
step tabbyapi:stack-r1              -f $R/Dockerfile.tabbyapi-prefbatch --build-arg BASE=tabbyapi:mixstate-r1 $R
step tabbyapi:slotfix-r1            -f $O/slotfix-r1/Dockerfile.box $O/slotfix-r1
step tabbyapi:hcfast-r1             -f $O/hcfast-r1/Dockerfile.box --build-arg BASE=tabbyapi:slotfix-r1 --build-arg MAX_JOBS=$J $O/hcfast-r1
step tabbyapi:stack-r2              -f $O/moefast-r1/Dockerfile.box --build-arg BASE=tabbyapi:hcfast-r1 --build-arg MAX_JOBS=$J $O/moefast-r1
step tabbyapi:stack-r3              -f $O/stack-r3/Dockerfile.box --build-arg BASE=tabbyapi:stack-r2 --build-arg INCLUDE="mf3 dg2" --build-arg DGV2_NVCC_DEFS= --build-arg SERIES_SHA=$(sha256sum $O/stack-r3/series/SERIES | cut -c1-64) --build-arg MAX_JOBS=$J $O/stack-r3
S3=tabbyapi:stack-r3
if (( n + 1 >= START )); then
  S3ID=$(podman image inspect $S3 --format '{{.Id}}')
  S3INC=$(podman image inspect $S3 --format '{{index .Config.Labels "local.stack.include"}}')
  S3DEF=$(podman image inspect $S3 --format '{{index .Config.Labels "local.stack.dgv2_defs"}}')
fi
step tabbyapi:stack-r3-rows32       -f $O/rows32-r4/Dockerfile.box --build-arg BASE=$S3 --build-arg BASE_ID="${S3ID:-unset}" --build-arg STACK_INCLUDE="${S3INC:-mf3 dg2}" --build-arg DGV2_NVCC_DEFS="${S3DEF:-}" --build-arg PATCH_SHA=$(sha256sum $O/rows32-r4/rows32-r4.patch | cut -c1-64) --build-arg MAX_JOBS=$J $O/rows32-r4
echo "=== chain done $(date -Is)" | tee -a $L/chain.log
