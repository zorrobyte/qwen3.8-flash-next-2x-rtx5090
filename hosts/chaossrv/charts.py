#!/usr/bin/env python3
"""Render the 2026-09-25 chaossrv benchmark charts (PNG) from hosts/chaossrv/results into hosts/chaossrv/charts.
The 27B points are the 2026-09-24 vLLM serving-bench results on the same box, entered as constants."""
import collections, csv, json, re, statistics as S
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
B = HERE / "results"
OUT = HERE / "charts"; OUT.mkdir(exist_ok=True)
HIS = HERE.parents[1] / "bench/results/2026-09-25-r719b-decode-curve/curve.tsv"

# validated reference palette, light mode (dataviz skill references/palette.md)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": 150,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
})


def med_curve(path):
    """probe.py log lines -> {conc: median decode_tps per stream}"""
    d = collections.defaultdict(list)
    for line in open(path):
        m = re.search(r"c=(\d+) run\d ctx~\d+: (\{.*\})", line)
        if m and "warm" not in line:
            d[int(m.group(1))].append(json.loads(m.group(2))["decode_tps"])
    return {c: S.median(v) for c, v in sorted(d.items())}


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords="offset points", va="center", fontsize=9, color=INK2)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("wrote", OUT / name)


# ---------- 1. decode curve vs adrienbrault R719b
ours = {k: med_curve(B / f"oc4500/{k}.log") for k in ("code", "prose")}
his = collections.defaultdict(dict)
for row in csv.DictReader(open(HIS), delimiter="\t"):
    his[row["kind"]][int(row["conc"])] = float(row["decode_agg"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, kind in zip(axes, ("code", "prose")):
    xs = list(ours[kind]); ya = [ours[kind][c] * c for c in xs]; yb = [his[kind][c] for c in xs]
    ax.plot(xs, ya, "-o", color=C1, label="chaossrv (tuned)", markeredgecolor=SURF, markeredgewidth=1.5)
    ax.plot(xs, yb, "-o", color=C2, label="adrienbrault R719b", markeredgecolor=SURF, markeredgewidth=1.5)
    up = ya[-1] >= yb[-1]; label_end(ax, xs, ya, f"{ya[-1]:.0f}", C1, 7 if up else -7); label_end(ax, xs, yb, f"{yb[-1]:.0f}", C2, -7 if up else 7)
    ax.set_title(f"{kind.capitalize()}: aggregate decode tok/s"); ax.set_xlabel("concurrent streams"); ax.set_xticks(xs); ax.set_ylim(0)
axes[0].set_ylabel("tok/s (sum over streams)"); axes[0].legend(loc="upper left")
fig.suptitle("Qwen3.8-Flash-Next EXL3 2.50bpw, 2× RTX 5090 — greedy, 1,024 forced tokens, ~110-token prompts", x=0.01, ha="left", fontsize=10, color=INK2)
save(fig, "01-decode-curve-vs-reference.png")

# ---------- 2. per-stream
fig, ax = plt.subplots(figsize=(8, 4))
for kind, col in (("code", C1), ("prose", C2)):
    xs = list(ours[kind]); ys = [ours[kind][c] for c in xs]
    ax.plot(xs, ys, "-o", color=col, label=f"{kind}", markeredgecolor=SURF, markeredgewidth=1.5)
    label_end(ax, xs, ys, f"{ys[-1]:.0f}", col)
    ax.annotate(f"{ys[0]:.0f}", (xs[0], ys[0]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK2)
ax.set_title("Per-stream decode tok/s (chaossrv, tuned)"); ax.set_xlabel("concurrent streams"); ax.set_ylabel("tok/s per stream")
ax.set_xticks(range(1, 9)); ax.set_ylim(0); ax.legend()
save(fig, "02-per-stream.png")

# ---------- 3. Flash-Next vs 27B pair (vLLM, one instance per card; 2026-09-24 CachyOS runs)
q27 = {"prose": {1: 232.1, 2: 234.5 + 232.4, 4: 711.5, 8: 707.0 + 698.5},
       "code": {1: 347.7, 2: 340.1 + 326.6, 4: 910.1, 8: 890.6 + 801.0}}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, kind in zip(axes, ("code", "prose")):
    xs = list(ours[kind]); ya = [ours[kind][c] * c for c in xs]
    ax.plot(xs, ya, "-o", color=C1, label="Flash-Next 125B MoE (both cards, one model)", markeredgecolor=SURF, markeredgewidth=1.5)
    xb = sorted(q27[kind]); yb = [q27[kind][c] for c in xb]
    ax.plot(xb, yb, "--s", color=C2, label="Qwen3.8-27B NVFP4 (vLLM, one per card; measured points)", markeredgecolor=SURF, markeredgewidth=1.5)
    up = ya[-1] >= yb[-1]; label_end(ax, xs, ya, f"{ya[-1]:.0f}", C1, 7 if up else -7); label_end(ax, xb, yb, f"{yb[-1]:.0f}", C2, -7 if up else 7)
    ax.set_title(f"{kind.capitalize()}: aggregate tok/s"); ax.set_xlabel("concurrent streams"); ax.set_xticks(range(1, 9)); ax.set_ylim(0)
axes[0].set_ylabel("tok/s (sum over streams)"); axes[0].legend(loc="upper left")
fig.suptitle("Different harnesses (27B: vLLM serving bench, thinking off; Flash-Next: probe.py greedy) — shape, not exact deltas", x=0.01, ha="left", fontsize=9, color=MUTED)
save(fig, "03-flashnext-vs-27b.png")

# ---------- 4/5. context sweep: prefill and decode at depth
rows = {k: [json.loads(l) for l in open(B / f"ctxsweep/{k}-c1.jsonl")] for k in ("code", "prose")}
fig, ax = plt.subplots(figsize=(8, 4))
for kind, col in (("code", C1), ("prose", C2)):
    pts = sorted((r["prompt_tokens"], r["prompt_tokens"] / r["ttft_s"]) for r in rows[kind] if r.get("ok") and r["prompt_tokens"] > 1000)
    ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], "-o", color=col, label=f"chaossrv {kind}", markeredgecolor=SURF, markeredgewidth=1.5)
ref = [(30.133, 9706), (60.014, 10381), (120.075, 10538), (199.844, 10636), (240.047, 10543)]
ax.plot([p[0] for p in ref], [p[1] for p in ref], "--D", color=C3, label="adrienbrault R580", markeredgecolor=SURF, markeredgewidth=1.5)
ax.set_title("Cold prefill rate vs prompt length (1 stream)"); ax.set_xlabel("prompt tokens (thousands)"); ax.set_ylabel("prefill tok/s"); ax.set_ylim(0); ax.legend(loc="lower right")
save(fig, "04-prefill.png")

fig, ax = plt.subplots(figsize=(8, 4))
for kind, col in (("code", C1), ("prose", C2)):
    pts = sorted((r["prompt_tokens"], r["decode_tps"]) for r in rows[kind] if r.get("ok"))
    ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], "-o", color=col, label=kind, markeredgecolor=SURF, markeredgewidth=1.5)
ax.set_title("Decode tok/s at depth (1 stream, 1,024 forced tokens after a cold prompt)"); ax.set_xlabel("prompt tokens (thousands)"); ax.set_ylabel("decode tok/s"); ax.set_ylim(0); ax.legend(loc="lower right")
ax.text(0.01, 0.04, "one request per point; the swings follow MTP draft acceptance on the generated text, not context length", transform=ax.transAxes, fontsize=8.5, color=MUTED)
save(fig, "05-decode-at-depth.png")

# ---------- 6. power timeline during context sweep
def power(path, t_from=None, t_to=None):
    d = collections.defaultdict(list)
    for r in csv.reader(open(path), skipinitialspace=True):
        if len(r) < 3: continue
        try: t = datetime.strptime(r[0][:19], "%Y/%m/%d %H:%M:%S")
        except ValueError: continue
        if t_from and t < t_from: continue
        if t_to and t > t_to: continue
        d[int(r[1])].append((t, float(r[2])))
    return d

pw = power(B / "ctxsweep/power.csv")
t0 = min(t for s in pw.values() for t, _ in s)
fig, ax = plt.subplots(figsize=(11, 3.8))
for g, col in ((0, C1), (1, C2)):
    ax.plot([(t - t0).total_seconds() / 60 for t, _ in pw[g]], [p for _, p in pw[g]], color=col, lw=1.2, label=f"GPU{g}")
ax.axhline(600, color=MUTED, ls="--", lw=1); ax.annotate("600 W limit", (0, 600), xytext=(4, 4), textcoords="offset points", fontsize=9, color=MUTED)
for g, col in ((0, C1), (1, C2)):
    t, p = max(pw[g], key=lambda x: x[1])
    ax.annotate(f"GPU{g} peak {p:.0f} W", ((t - t0).total_seconds() / 60, p), xytext=(8, 6 if g == 0 else -14), textcoords="offset points", fontsize=9, color=INK2)
ax.set_title("GPU power during the context sweep (cold prefill up to 180k tokens + decode)"); ax.set_xlabel("minutes"); ax.set_ylabel("W"); ax.set_ylim(0, 650); ax.legend(loc="upper right", ncols=2)
save(fig, "06-power-context-sweep.png")

# ---------- 7. peak power by workload
peaks = [("Decode, stock\n(575 W, no OC)", 297.4, 281.3, 574.5), ("Decode 1–8 streams\n(600 W + OC)", 302.8, 285.0, 583.7), ("Long-prompt prefill\n(600 W + OC)", 548.8, 496.6, 1045.4)]
fig, ax = plt.subplots(figsize=(8, 4))
import numpy as np
x = np.arange(len(peaks)); w = 0.36
b0 = ax.bar(x - w / 2 - 0.01, [p[1] for p in peaks], w, color=C1, label="GPU0")
b1 = ax.bar(x + w / 2 + 0.01, [p[2] for p in peaks], w, color=C2, label="GPU1")
for bars in (b0, b1):
    for r in bars: ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK2)
for i, p in enumerate(peaks):
    ax.annotate(f"both at once: {p[3]:.0f} W", (i, 650), ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.axhline(600, color=MUTED, ls="--", lw=1); ax.annotate("600 W limit", (-0.5, 600), xytext=(4, -12), textcoords="offset points", ha="left", fontsize=9, color=MUTED)
ax.set_xticks(x, [p[0] for p in peaks]); ax.set_ylabel("peak W (nvidia-smi, 250–500 ms)"); ax.set_ylim(0, 720); ax.grid(axis="x", visible=False)
ax.set_title("Peak GPU power by workload (per card, and both cards at the same instant)"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "07-power-peaks.png")

# ---------- 8. tuning log (1 stream)
steps = [("stock host", 239, 245), ("EPP performance", 247, 245), ("pin to one CCD", 234, 241), ("NVMe tier off", 239, 246),
         ("P2P driver", 229, 245), ("C2/C3 off", 238, 249), ("600 W + mem +4500", 242, 254), ("reference R719b", 298, 279)]
fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(steps))[::-1]; h = 0.36
ax.barh(y + h / 2 + 0.01, [s[1] for s in steps], h, color=C1, label="code")
ax.barh(y - h / 2 - 0.01, [s[2] for s in steps], h, color=C2, label="prose")
for yy, s in zip(y, steps):
    ax.annotate(f"{s[1]}", (s[1], yy + h / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.annotate(f"{s[2]}", (s[2], yy - h / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, [s[0] for s in steps]); ax.set_xlim(0, 330); ax.set_xlabel("decode tok/s, 1 stream"); ax.grid(axis="y", visible=False)
ax.set_title("Tuning steps: 1-stream decode tok/s"); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
save(fig, "08-tuning-log.png")

# ---------- 9. P2P
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
labels = ["stock driver\n(host-staged)", "P2P driver\n(now)"]
bw = [22.15, 28.68]; lat = [9.5, 1.17]
for ax, vals, title, unit, cols in ((axes[0], bw, "GPU→GPU copy bandwidth", "GB/s", (MUTED, C1)), (axes[1], lat, "Small-copy latency (4 B)", "µs", (MUTED, C1))):
    bars = ax.bar(labels, vals, 0.5, color=cols)
    for r in bars: ax.annotate(f"{r.get_height():g} {unit}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals) * 1.25)
save(fig, "09-p2p.png")
