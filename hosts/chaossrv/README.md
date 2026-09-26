# chaossrv: reproducing the stack on a second 2× RTX 5090 box

A second machine running this repo's `tabbyapi:stack-r3-rows32` with the same checkpoint
(`r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw`), built from source with rootless podman.

## Hardware and host

- 2× RTX 5090 (Zotac `cuda:0`, ASUS TUF `cuda:1`), PCIe 5.0 x8/x8, topology PHB
- AMD Ryzen 9 9950X3D (2 CCDs, V-cache on CCD0), 91 GB DDR5
- CachyOS, kernel 7.2.7 (clang build), NVIDIA 615.71.09 open modules, rootless podman + CDI
- Display manager stopped while serving (both cards at 2 MiB used before boot)

## Fixes needed to build the chain (upstream issue #1)

Three breaks were found here: `hc-mix-v2-r2.patch` depended on an unpublished round-1 patch; the `ngram-prefetch-r1`
manifest expected a `prefill_pipeline.py` hash that no published layer produced (the served `-mtpfix2` layer also ran
an unpublished `memfix.py`); and `-mixstate` reused stale `/tmp/build` objects. The first images here used local
workarounds (the refbase files, a rebased manifest entry, a `/tmp/build` cleanup). Upstream `07141f3` publishes the
round-1 patch and `memfix.py`, clears `/tmp/build` in every native rebuild, and adds `docker/build-chain.sh`; this
branch merges it and drops the workarounds.

Build: `DOCKER=hosts/chaossrv/podman-docker.sh bash docker/build-chain.sh` (upstream script since `07141f3`, 35 layers). Serve:
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

## Charts (2026-09-25)

Rendered by `hosts/chaossrv/charts.py` from `results/` into `charts/`. Colors follow one fixed order per chart; every
chart with two or more series carries a legend.

### Decode throughput against R719b

![Aggregate decode tok/s against concurrency, chaossrv and R719b, code and prose](charts/01-decode-curve-vs-reference.png)

Aggregate decode at 1 to 8 streams, greedy, 1,024 forced tokens, 118 / 106-token prompts (`results/oc4500/`), with
R719b's `curve.tsv` from this repository. Against R719b the aggregate is −18.8 % (code) and −8.7 % (prose) at 1
stream, −0.4 % and +7.1 % at 2, −4.9 to −9.2 % (code) and −3.7 to −7.7 % (prose) at 3 to 6, and −1.6 to −2.2 % at 7
and 8.

![Per-stream decode tok/s against concurrency, code and prose](charts/02-per-stream.png)

### Against the Qwen3.8-27B NVFP4 pair on the same box

![Aggregate tok/s against concurrency, Flash-Next layer split and 27B one vLLM per card](charts/03-flashnext-vs-27b.png)

The 27B points are the 2026-09-24 vLLM serving-bench results on this box (thinking off): 1 stream on one card, 2
streams as one per card, 4 streams on one card, 8 streams as four per card. The two harnesses differ, so the chart
shows the shape of the two curves; single-digit percentage differences between them carry no information. At 8
streams the 27B pair reaches 1,406 prose / 1,692 code against 840 / 830, since each card serves its own model while
the layer split alternates the cards on every step.

### Prefill and decode at depth

![Cold prefill tok/s against prompt tokens, chaossrv code and prose with R580](charts/04-prefill.png)

Cold prefill (salted filler, 1 stream, `results/ctxsweep/`) reaches 9,849 to 10,662 t/s from 43k tokens (code) and
24k tokens (prose) upward, against R580's 9,706 to 10,636 t/s. R442 measured the two-card prefill pipeline at 36 %
(30k) and 41 % (120k) less prefill time than the same split without it.

![Decode tok/s at depth against prompt tokens, code and prose](charts/05-decode-at-depth.png)

One request per point. Decode stays between 225 and 346 t/s from 6k to 180k tokens of context; the variation follows
MTP draft acceptance on the generated text.

### Power

![GPU power in watts over the context sweep, GPU0 and GPU1, with the 600 W limit](charts/06-power-context-sweep.png)

nvidia-smi `power.draw` at 500 ms during the context sweep. Prefill bursts hold both cards at 480 to 549 W together;
decode holds each card near 300 W.

![Peak power per card and both cards at the same instant, by workload](charts/07-power-peaks.png)

Peak over every logged run on 2026-09-25: 1,045 W with both cards at the same instant (GPU0 548.8 W, GPU1 496.6 W)
during cold prefill of 150k to 180k-token prompts; 584 W during the 1 to 8-stream decode curve; 575 W during decode at
the stock 575 W limit without the memory offset. No card reached its power limit and no power-cap clock limiter was
active in the samples. `power.draw` is an averaged reading, so transients shorter than a sample are not in these
figures.

### Tuning and P2P

![1-stream decode tok/s per tuning step, code and prose, with R719b](charts/08-tuning-log.png)

![GPU-to-GPU copy bandwidth and 4-byte copy latency, stock driver and P2P driver](charts/09-p2p.png)

Stock 615.71.09 stages GPU-to-GPU copies through host memory (22.15 GB/s, 9.5 µs); the aikitoria `615.71.09-p2p`
modules give 28.68 GB/s and 1.17 µs. Decode rates with and without P2P are within the run-to-run spread.
