#!/usr/bin/env python3
"""chaossrv system monitor: samples the GPUs (nvidia-smi), the host (/proc, /sys) and the flashnext
TabbyAPI server (/live, /metrics, container log) once a second, keeps 10 minutes of history and serves
a dashboard on 127.0.0.1:18100 (published on the tailnet via `tailscale serve --https=8443`)."""
import glob, json, os, re, subprocess, threading, time, urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 18100
TABBY = "http://127.0.0.1:8022"
CONTAINER = "flashnext"
HISTORY = 600
HERE = Path(__file__).parent

# ---------------------------------------------------------------- GPUs
GPU_FIELDS = ["index", "name", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
              "power.draw", "power.limit", "power.max_limit", "temperature.gpu", "clocks.sm", "clocks.mem",
              "clocks.max.sm", "clocks.max.mem", "fan.speed", "pcie.link.gen.current", "pcie.link.width.current",
              "pstate", "clocks_event_reasons.active"]
REASONS = {0x1: "idle", 0x2: "app clocks", 0x4: "sw power cap", 0x8: "hw slowdown", 0x10: "sync boost",
           0x20: "sw thermal", 0x40: "hw thermal", 0x80: "hw power brake", 0x100: "display clocks"}


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    res = []
    for row in out.strip().splitlines():
        f = [x.strip() for x in row.split(",")]
        if len(f) < len(GPU_FIELDS):
            continue
        d = dict(zip(GPU_FIELDS, f))
        try:
            mask = int(d["clocks_event_reasons.active"], 16)
        except ValueError:
            mask = 0
        res.append({
            "index": int(d["index"]), "name": d["name"].replace("NVIDIA GeForce ", ""),
            "util": num(d["utilization.gpu"]), "mem_util": num(d["utilization.memory"]),
            "mem_used": num(d["memory.used"]), "mem_total": num(d["memory.total"]),
            "power": num(d["power.draw"]), "power_limit": num(d["power.limit"]), "power_max": num(d["power.max_limit"]),
            "temp": num(d["temperature.gpu"]), "sm_clock": num(d["clocks.sm"]), "mem_clock": num(d["clocks.mem"]),
            "sm_max": num(d["clocks.max.sm"]), "mem_max": num(d["clocks.max.mem"]), "fan": num(d["fan.speed"]),
            "pcie": f'Gen{d["pcie.link.gen.current"]} x{d["pcie.link.width.current"]}', "pstate": d["pstate"],
            "reasons": [n for b, n in REASONS.items() if mask & b and b != 0x1],
        })
    return res


def p2p_status():
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-p2p", "r"], capture_output=True, text=True, timeout=5).stdout
        return "OK" if re.search(r"\bOK\b", out) else "off"
    except Exception:
        return "?"


# ---------------------------------------------------------------- host
def read(path, default=""):
    try:
        return Path(path).read_text()
    except OSError:
        return default


class Rates:
    """Turns monotonically increasing counters into per-second rates."""
    def __init__(self):
        self.prev = {}

    def rate(self, key, value, now):
        p = self.prev.get(key)
        self.prev[key] = (value, now)
        if p is None or now <= p[1] or value < p[0]:
            return None
        return (value - p[0]) / (now - p[1])


rates = Rates()


def cpu_times():
    out = {}
    for line in read("/proc/stat").splitlines():
        if line.startswith("cpu"):
            f = line.split()
            vals = list(map(int, f[1:9]))
            idle = vals[3] + vals[4]
            out[f[0]] = (sum(vals), idle)
    return out


_prev_cpu = {}


def cpu():
    global _prev_cpu
    cur = cpu_times()
    util = {}
    for k, (tot, idle) in cur.items():
        if k in _prev_cpu:
            dt, di = tot - _prev_cpu[k][0], idle - _prev_cpu[k][1]
            util[k] = 100.0 * (1 - di / dt) if dt > 0 else 0.0
    _prev_cpu = cur
    cores = sorted((k for k in util if k != "cpu"), key=lambda k: int(k[3:]))
    freqs = []
    for k in cores:
        f = num(read(f"/sys/devices/system/cpu/{k}/cpufreq/scaling_cur_freq").strip())
        freqs.append(round(f / 1000) if f else None)
    load = read("/proc/loadavg").split()[:3]
    return {"total": util.get("cpu"), "cores": [round(util[k], 1) for k in cores], "freq": freqs,
            "load": [num(x) for x in load], "epp": read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").strip()}


def memory():
    m = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return {"total": m.get("MemTotal"), "available": m.get("MemAvailable"), "cached": m.get("Cached", 0) + m.get("Buffers", 0),
            "swap_total": m.get("SwapTotal"), "swap_used": m.get("SwapTotal", 0) - m.get("SwapFree", 0)}


def temps():
    res = []
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read(f"{hw}/name").strip()
        if name not in ("k10temp", "nvme", "spd5118"):
            continue
        for inp in sorted(glob.glob(f"{hw}/temp*_input")):
            label = read(inp.replace("_input", "_label")).strip() or "temp"
            v = num(read(inp).strip())
            if v is not None:
                res.append({"chip": name, "label": label, "c": round(v / 1000, 1)})
    return res


DISK_RE = re.compile(r"^(nvme\d+n\d+|sd[a-z]+)$")


def disks(now):
    res = []
    for line in read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) < 14 or not DISK_RE.match(f[2]):
            continue
        r = rates.rate("dr" + f[2], int(f[5]) * 512, now)
        w = rates.rate("dw" + f[2], int(f[9]) * 512, now)
        res.append({"dev": f[2], "read": r, "write": w})
    return res


def net(now):
    res = []
    for line in read("/proc/net/dev").splitlines()[2:]:
        name, data = line.split(":", 1)
        name = name.strip()
        if name == "lo" or name.startswith(("veth", "podman", "cni")):
            continue
        f = data.split()
        rx = rates.rate("nr" + name, int(f[0]), now)
        tx = rates.rate("nt" + name, int(f[8]), now)
        res.append({"iface": name, "rx": rx, "tx": tx})
    return res


def cstates():
    dis = read("/sys/devices/system/cpu/cpu0/cpuidle/state2/disable").strip()
    return "C2+ off" if dis == "1" else "default"


# ---------------------------------------------------------------- flashnext / TabbyAPI
def get_json(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


METRIC = re.compile(r'^(tabby_[a-z_]+)(\{[^}]*\})?\s+(\S+)$')


def tabby_counters():
    try:
        with urllib.request.urlopen(TABBY + "/metrics", timeout=2) as r:
            text = r.read().decode()
    except Exception:
        return None
    out = {}
    for line in text.splitlines():
        m = METRIC.match(line)
        if m and not m.group(1).endswith(("_bucket",)):
            out[m.group(1)] = out.get(m.group(1), 0) + (num(m.group(3)) or 0)
    return out


# Completed-request lines from the container log (wrapped over several lines by rich)
REQ_RE = re.compile(
    r"#(?P<n>\d+) (?P<kind>[\w/]+(?: \(stream\))?): (?P<gen>[\d,]+) tokens generated at (?P<tps>[\d.,]+) T/s"
    r".*?prompt (?P<prompt>[\d,]+) tokens, (?P<cached>none|[\d,]+%?) cached"
    r"(?:.*?queued (?P<queue>[\d.]+) s)?.*?first token (?P<ttft>[\d.]+) s, total (?P<total>[\d.]+) s"
    r"(?:.*?draft (?P<acc>[\d,]+)/(?P<drafted>[\d,]+) accepted)?(?:.*?·\s*(?P<finish>[a-z_ ]+reached|stop[^·]*|eos[^·]*|loop detected))?")
recent = deque(maxlen=40)


def log_follower():
    while True:
        try:
            p = subprocess.Popen(["podman", "logs", "-f", "--since", "10m", CONTAINER], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            buf = ""
            for line in p.stdout:
                if re.match(r"^\d\d:\d\d:\d\d\.\d{3} ", line):
                    handle_entry(buf)
                    buf = line
                else:
                    buf += " " + line.strip()
            handle_entry(buf)
        except Exception:
            pass
        time.sleep(5)


def handle_entry(entry):
    entry = re.sub(r"\s+", " ", entry)
    m = REQ_RE.search(entry)
    if not m:
        return
    g = m.groupdict()
    i = lambda s: int(s.replace(",", "")) if s and s != "none" else 0
    prompt = i(g["prompt"])
    c = g["cached"] or "none"
    cached = round(prompt * float(c[:-1]) / 100) if c.endswith("%") else i(c)
    rec = {"time": entry[:12], "n": int(g["n"]), "kind": g["kind"], "gen": i(g["gen"]),
           "tps": float(g["tps"].replace(",", "")), "prompt": prompt, "cached": cached,
           "queue": num(g["queue"]), "ttft": num(g["ttft"]), "total": num(g["total"]),
           "accept": (i(g["acc"]) / i(g["drafted"])) if g["drafted"] and i(g["drafted"]) else None,
           "finish": (g["finish"] or "").strip()}
    if not any(r["n"] == rec["n"] and r["time"] == rec["time"] for r in recent):
        recent.appendleft(rec)


# ---------------------------------------------------------------- sampler
state = {"history": deque(maxlen=HISTORY), "latest": None, "static": {}}
lock = threading.Lock()


def sample_loop():
    last_static = 0
    while True:
        t0 = time.time()
        now = time.monotonic()
        live = get_json(TABBY + "/live")
        counters = tabby_counters()
        s = {"t": t0, "gpus": gpus(), "cpu": cpu(), "mem": memory(), "temps": temps(),
             "disks": disks(now), "net": net(now), "live": live, "up": live is not None}
        if counters:
            s["counters"] = counters
            s["prompt_rate"] = rates.rate("prompt_tok", counters.get("tabby_prompt_tokens_total", 0), now)
        if t0 - last_static > 30:
            state["static"] = {"p2p": p2p_status(), "cstates": cstates(),
                               "uptime": num(read("/proc/uptime").split()[0]), "host": os.uname().nodename,
                               "kernel": os.uname().release}
            last_static = t0
        with lock:
            state["latest"] = s
            state["history"].append({
                "t": t0,
                "tps": (live or {}).get("tps") or 0,
                "active": len((live or {}).get("jobs") or []),
                "pool": ((live or {}).get("cache") or {}).get("used_tokens"),
                "gpu_power": [g["power"] for g in s["gpus"]],
                "gpu_util": [g["util"] for g in s["gpus"]],
                "gpu_temp": [g["temp"] for g in s["gpus"]],
                "cpu": s["cpu"]["total"],
                "mem_used": (s["mem"]["total"] or 0) - (s["mem"]["available"] or 0),
            })
        time.sleep(max(0.05, 1.0 - (time.time() - t0)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            with lock:
                body = {"latest": state["latest"], "static": state["static"], "recent": list(recent)}
            self.send(200, json.dumps(body).encode(), "application/json")
        elif path == "/api/history":
            with lock:
                body = list(state["history"])
            self.send(200, json.dumps(body).encode(), "application/json")
        else:
            self.send(404, b"not found", "text/plain")


if __name__ == "__main__":
    threading.Thread(target=sample_loop, daemon=True).start()
    threading.Thread(target=log_follower, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
