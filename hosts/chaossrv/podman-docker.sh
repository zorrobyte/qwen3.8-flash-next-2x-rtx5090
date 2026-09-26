#!/usr/bin/env bash
# docker-compatible shim so docker/build-chain.sh runs on rootless podman.
case "$1 $2" in
  "buildx version") podman version --format 'podman {{.Client.Version}}'; exit 0;;
  "buildx inspect") echo "Driver: docker"; exit 0;;
esac
if [ "$1" = build ]; then
  shift; args=()
  for a in "$@"; do [ "$a" = "--progress=plain" ] || args+=("$a"); done
  exec podman build --network host --format docker "${args[@]}"
fi
exec podman "$@"
