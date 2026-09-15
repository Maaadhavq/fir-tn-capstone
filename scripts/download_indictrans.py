"""Fetch the configured Tamil->English translation model into the local HF cache.

    python scripts/download_indictrans.py

Reads `translate.model` from configs/pipeline.yaml. For IndicTrans2 (gated) run
`huggingface-cli login` first and accept the terms on the model page; its
toolkit is installed automatically. Local-only thereafter (HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import pipeline  # noqa: E402

MODEL_ID = pipeline().get("translate", {}).get("model", "Helsinki-NLP/opus-mt-dra-en")


def main() -> int:
    if "indictrans" in MODEL_ID.lower():
        print("==> IndicTransToolkit")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "IndicTransToolkit"])

    print(f"==> {MODEL_ID}")
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6
    print(f"    cached at {path}  ({size:.0f} MB)")

    print("==> smoke test")
    from fir.translate.indictrans import IndicTranslator

    tr = IndicTranslator()
    out = tr.translate("என் போன் திருடப்பட்டது. ஐம்பதாயிரம் ரூபாய் மதிப்பு.", force=True)
    print(f"    [{tr.resolved_device}] {out.target!r}  ({out.seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
