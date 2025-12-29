# SzCORE submission (epilepsybenchmarks.com)

This folder contains a buildable Docker image definition for SzCORE submission.

## Build

From repo root:

```bash
docker build -f deploy/szcore/Dockerfile -t brain-go-brr-szcore:local .
```

## CI-format check (mirrors SzCORE PR CI)

SzCORE PR CI runs the container on a CPU-only GitHub Actions runner and checks the TSV header.

```bash
mkdir -p /tmp/szcore_out
curl -fsSL -L https://raw.githubusercontent.com/esl-epfl/szcore/main/tests/data/unipolar.edf -o /tmp/unipolar.edf
docker run --rm \
  -v /tmp:/data \
  -v /tmp/szcore_out:/output \
  -e INPUT=unipolar.edf \
  -e OUTPUT=brain_go_brr.tsv \
  brain-go-brr-szcore:local
head -n1 /tmp/szcore_out/brain_go_brr.tsv
```

Expected header:

```text
onset	duration	eventType	confidence	channels	dateTime	recordingDuration
```

## Notes

- The container uses a CPU heuristic fallback when no GPU is present (to satisfy PR CI).
- When a GPU is available, it loads the FLA Exp4 checkpoint and runs full Brain-Go-Brr inference.

