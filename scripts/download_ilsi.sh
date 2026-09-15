#!/usr/bin/env bash
# Download the ILSI statute-identification corpus from Zenodo (10.5281/zenodo.6053791).
# Open access, no request form. ~512 MB for the files we actually use.
#
# Deliberately SKIPPED:
#   ils2v.bin        (2.25 GB) -- pretrained statute embeddings, only needed to
#                                 reproduce LeSICiN's graph model, not our baseline.
#   best_model.pt    (40 MB)   -- LeSICiN checkpoint.
#   citation_network.json      -- statute citation graph, for the later RAG phase.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/data/ilsi"
BASE="https://zenodo.org/records/6053791/files"

mkdir -p "$DEST"
for f in secs.jsonl label_vocab.json type_map.json dev.jsonl test.jsonl train.jsonl; do
  if [[ -s "$DEST/$f" ]]; then
    echo "==> $f already present ($(du -h "$DEST/$f" | cut -f1)), skipping"
    continue
  fi
  echo "==> downloading $f"
  curl -fL --retry 3 --retry-delay 5 -o "$DEST/$f.part" "$BASE/$f?download=1"
  mv "$DEST/$f.part" "$DEST/$f"
done

echo
echo "==> ILSI ready:"
du -h "$DEST"/*.jsonl "$DEST"/*.json | sort -h
