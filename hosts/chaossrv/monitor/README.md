# Host monitor

`server.py` (Python stdlib) samples `nvidia-smi`, `/proc` and `/sys`, the flashnext server's `/live` and `/metrics`
endpoints and its container log once a second, keeps 10 minutes of history and serves `index.html` on
127.0.0.1:18100. `/live` comes from the `live-status-r1` overlay (`docker/overlays/live-status-r1`): active jobs with
stage, prefill progress, generated tokens and tokens per second, plus the generator's page-pool statistics.

The container log reports prompt-cache reuse as a percentage (`prompt 103,173 tokens, 100% cached`) or `none cached`;
the parser converts the percentage to tokens.
