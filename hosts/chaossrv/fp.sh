#!/usr/bin/env bash
# adrienbrault G1 c1 greedy fingerprint (expected f4add302e176d78e for the served stack since R548)
M=${MODEL:-Qwen3.8-Flash-Next-EXL3-2.50bpw}
P="Write a Python function that parses an ISO-8601 duration string into a datetime.timedelta, with tests. No explanation."
curl -s -m 900 http://127.0.0.1:8022/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys;print(json.dumps({"model":sys.argv[1],"temperature":0,"max_tokens":256,"min_tokens":256,"messages":[{"role":"user","content":sys.argv[2]}]}))' "$M" "$P")" > /tmp/claude-1000/fp.json
python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; t=(m.get("content") or "")+"|"+(m.get("reasoning_content") or ""); print(hashlib.sha256(t.encode()).hexdigest()[:16])' /tmp/claude-1000/fp.json
