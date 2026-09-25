"""Live snapshot of the generator for dashboards (chaossrv overlay live-status-r1).

Reads the state TabbyAPI already keeps for its console status display: one JobStatus per active
request (stage, prefill progress, generated tokens, windowed tokens/s) and the generator's cache
statistics. Read-only; nothing here touches generation.
"""
import time

from common import model
from common.status_display import StatusDisplay, status_display


def live_snapshot() -> dict:
    now = time.monotonic()
    jobs = []
    for request_id, job in list(status_display.jobs.items()):
        jobs.append({
            "id": str(request_id)[-8:],
            "label": job.label,
            "stage": job.stage,
            "prompt_tokens": job.prompt_tokens,
            "cached_tokens": job.cached_tokens,
            "prefill_tokens": job.prefill_tokens,
            "gen_tokens": job.gen_tokens,
            "tps": job.tokens_per_second(),
            "age_s": round(now - job.created, 2),
        })
    container = model.container
    loaded = bool(container and getattr(container, "loaded", False))
    return {
        "loaded": loaded,
        "model": getattr(container, "model_name", None) if container else None,
        "max_batch_size": getattr(container, "max_batch_size", None) if container else None,
        "max_seq_len": getattr(container, "max_seq_len", None) if container else None,
        "cache": StatusDisplay._cache_stats() if loaded else None,
        "completed": status_display.completed,
        "jobs": jobs,
        "tps": sum(j["tps"] or 0 for j in jobs),
    }
