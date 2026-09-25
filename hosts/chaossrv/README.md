# chaossrv: reproducing the stack on a second 2× RTX 5090 box

A second machine running this repo's `tabbyapi:stack-r3-rows32` with the same checkpoint
(`r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw`), built from source with rootless podman.

## Hardware and host

- 2× RTX 5090 (Zotac `cuda:0`, ASUS TUF `cuda:1`), PCIe 5.0 x8/x8, topology PHB
- AMD Ryzen 9 9950X3D (2 CCDs, V-cache on CCD0), 91 GB DDR5
- CachyOS, kernel 7.2.7 (clang build), NVIDIA 615.71.09 open modules, rootless podman + CDI
- Display manager stopped while serving (both cards at 2 MiB used before boot)

## Fixes needed to build the chain (upstream issue #1)

1. **`-hcmix2`**: `hc-mix-v2-r2.patch` is relative to an unpublished r1 V2 mixer; every hunk fails on stock
   ExLlamaV3 v1.5.0. `docker/Dockerfile.tabbyapi-hcmix2-refbase` installs the published full files from
   `overlays/refbase/files/` instead (refbase overwrites these files later in the chain anyway).
2. **`ngram-prefetch-r1`**: the manifest's baseline hash for `generator/prefill_pipeline.py` (`7f6e4b34…`) is produced
   by no published layer; the published chain has `1a5af53b…`. That file's hunk is comment-only, so the manifest entry
   is rebased (baseline `1a5af53b…`, overlay `e5e1389e…`); the other three files still match exactly.
3. **`Dockerfile.tabbyapi-mixstate`**: reused `/tmp/build` objects from earlier layers. The decode-kernels-r2 payload is
   COPY'd with checkout mtimes older than those objects, so ninja skipped `blocksparse_mlp.cpp` and the extension failed
   to import (`undefined symbol: exl3_moe_coop_run(...)` without the `cudaEvent_t` argument). Now clears `/tmp/build`.

Build: `docker/build-chain-podman.sh` (35 layers, about 45 minutes on 32 threads). Serve:
`scripts/launch-flashnext-podman.sh` (`FN_ROOT`, `FN_MODELS`).

## Numbers (2026-09-25)

`hosts/chaossrv/curve.sh`: `bench/probe.py --distinct`, greedy, 1,024 forced tokens, 1 warm-up + 3 rounds, NVMe tier
off, same prompts as R719b. Decode tok/s after the first token, median per stream; aggregate = per-stream × streams.

Tuned host (`host-tune.sh`: EPP performance, C2/C3 off, 600 W, +4500 memory offset), P2P driver loaded:

| streams | code per stream | prose per stream | code agg | prose agg | R719b code agg | R719b prose agg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 241.7 | 254.3 | 242 | 254 | 298 | 279 |
| 2 | 206.5 | 220.3 | 413 | 441 | 415 | 412 |
| 3 | 163.7 | 167.4 | 491 | 502 | 541 | 542 |
| 4 | 148.1 | 148.8 | 592 | 595 | 632 | 643 |
| 5 | 131.3 | 128.1 | 656 | 640 | 691 | 694 |
| 6 | 115.2 | 118.0 | 691 | 708 | 727 | 735 |
| 7 | 110.2 | 110.7 | 771 | 775 | 784 | 792 |
| 8 | 103.8 | 105.0 | 830 | 840 | 845 | 853 |

## What moved the 1-stream rate (c1 code / prose, tok/s)

| change | c1 | notes |
| --- | --- | --- |
| stock host, served launcher | 239 / 245 | 11.1 ms per decode step, 2.64 tokens/step (R719b: 10.0 ms, 2.99) |
| EPP `performance` | 247 / 245 | |
| container pinned to CCD0 or CCD1 | 234 / 241 | worse either way; left unpinned |
| NVMe tier off | 239 / 246 | no effect at these prompt lengths |
| coop autotune cache regenerated | 234 / 253 | changes greedy output (fingerprint `d589edfb…` to `5cd59025…`) |
| aikitoria 615.71.09-p2p modules (P2P OK, 1.17 µs) | 229 / 245 | no decode effect |
| C2/C3 off | 238 / 249 | 10.85 ms per step |
| 600 W + memory offset +4500 | 242 / 254 | fingerprint stable over 4 runs |

## Open: the 1-stream gap

Of the remaining gap to R719b at one stream, about half is MTP acceptance (2.64 vs 2.99 tokens per step on code) and
the rest is step time (10.85 vs 10.0 ms). The c1 greedy fingerprint is `5cd590252f16ceaf` here against the served
`f4add302e176d78e`. Regenerating the coop autotune cache changed it, so autotune choices (which differ per box)
change the numerics and therefore which drafts are accepted on a given prompt.

## Context sweep (2026-09-25, tuned host)

`hosts/chaossrv/ctxsweep.sh`: cold salted `--unique` filler per depth, 1 stream, 1,024 forced greedy tokens; one request
per cell, so decode varies with how predictable the generated text is (MTP acceptance). Raw: `results/ctxsweep/`.

| prompt tokens (code / prose) | prefill t/s (code / prose) | decode t/s (code / prose) | R580 prefill |
| --- | --- | --- | --- |
| 118 / 106 | n/a | 243 / 253 | |
| 11,145 / 6,102 | 4,103 / 5,478 | 238 / 235 | |
| 21,545 / 12,108 | 8,015 / 7,456 | 312 / 338 | |
| 43,130 / 24,167 | 9,849 / 9,072 | 304 / 225 | 9,706 @ 30k |
| 86,241 / 48,061 | 10,034 / 9,865 | 260 / 346 | 10,381 @ 60k |
| 172,343 / 96,132 | 10,376 / 10,335 | 274 / 232 | 10,538 @ 120k |
| n/a / 149,679 | n/a / 10,522 | n/a / 238 | 10,636 @ 200k |
| n/a / 180,259 | n/a / 10,662 | n/a / 235 | 10,543 @ 240k |

Code at 200k/240k filler exceeds the 262,144-token window (≈270k/320k tokens) and is rejected. 4 streams, code: 148.2
per stream at 43k, 139.9 at 172k. GPU power over the sweep (samples > 100 W): 355 / 325 W mean, 549 / 497 W peak.
