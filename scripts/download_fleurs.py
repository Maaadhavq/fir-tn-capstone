"""Fetch the FLEURS Tamil (ta_in) split used for the ASR baseline.

Uses `huggingface_hub.hf_hub_download` rather than `datasets.load_dataset`
because `datasets` depends on pyarrow, which Smart App Control blocks on the
Windows host (IMPLEMENTATION.md §1). We pull the raw tar + tsv and unpack them
ourselves -- no parquet involved.

    python scripts/download_fleurs.py --split test

Sizes: test.tar.gz 406.6 MB, dev.tar.gz 238.9 MB.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPO_ID = "google/fleurs"
CONFIG = "ta_in"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument("--dest", default=str(REPO_ROOT / "data" / "fleurs"))
    ap.add_argument(
        "--keep-archive",
        action="store_true",
        help="keep the .tar.gz after extraction (default: delete to save disk)",
    )
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    dest = Path(args.dest)
    audio_dir = dest / CONFIG / "audio" / args.split
    tsv_dest = dest / CONFIG / f"{args.split}.tsv"
    dest.mkdir(parents=True, exist_ok=True)
    tsv_dest.parent.mkdir(parents=True, exist_ok=True)

    # --- transcripts ---
    print(f"==> {CONFIG}/{args.split}.tsv")
    tsv = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/{CONFIG}/{args.split}.tsv",
    )
    tsv_dest.write_bytes(Path(tsv).read_bytes())

    # --- audio ---
    if audio_dir.exists() and any(audio_dir.iterdir()):
        n = sum(1 for _ in audio_dir.glob("*.wav"))
        print(f"==> audio already extracted ({n} clips), skipping")
    else:
        print(f"==> {CONFIG}/audio/{args.split}.tar.gz  (this is the big one)")
        tar_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/{CONFIG}/audio/{args.split}.tar.gz",
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> extracting to {audio_dir}")
        with tarfile.open(tar_path, "r:gz") as tf:
            # The archive nests everything under <split>/; flatten it so the
            # loader can resolve file_name from the tsv directly.
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.endswith(".wav"):
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                (audio_dir / name).write_bytes(src.read())

        if not args.keep_archive:
            # hf_hub_download caches under ~/.cache/huggingface; the blob is
            # what actually costs 400 MB.
            try:
                Path(tar_path).unlink()
                print("==> removed cached archive")
            except OSError:
                pass

    n = sum(1 for _ in audio_dir.glob("*.wav"))
    print(f"\n==> FLEURS {CONFIG}/{args.split} ready: {n} clips in {audio_dir}")
    print(f"    transcripts: {tsv_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
