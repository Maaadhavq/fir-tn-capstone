"""Tamil -> English translation quality on FLEURS/FLoRes parallel sentences.

    python -m harness.run_translation_eval            # 100 sentences
    python -m harness.run_translation_eval --n 336    # all of them
    python -m harness.run_translation_eval --model facebook/nllb-200-distilled-600M

FLEURS is built on FLoRes-200, so the Tamil (`ta_in`) and English (`en_us`)
test TSVs share sentence IDs -- a free Ta->En parallel set of 336 pairs. This
scores the configured translator (configs/pipeline.yaml `translate.model`) on
it with chrF (the metric that behaves for morphologically rich source languages)
and BLEU.

Why it matters: the statute classifier is English-only, so translation quality
is an upper bound on statute-ID quality for Tamil complaints. A weak translator
does not just lose words -- it loses the words the classifier keys on.

FLoRes is Wikipedia-register prose, not police complaints; treat these numbers
as a model-comparison signal, not a deployment estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import ARTIFACT_DIR, DATA_DIR  # noqa: E402
from fir.translate.indictrans import IndicTranslator  # noqa: E402


def _tsv_transcriptions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) > 3 and row[0] not in out:
                out[row[0]] = row[3].strip()
    return out


def load_pairs(limit: int | None = None) -> list[tuple[str, str, str]]:
    ta = _tsv_transcriptions(DATA_DIR / "fleurs" / "ta_in" / "test.tsv")
    en_path = DATA_DIR / "fleurs" / "en_us" / "test.tsv"
    if not en_path.exists():   # 0.8 MB; fetch on demand rather than commit it
        import shutil

        from huggingface_hub import hf_hub_download

        en_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download("google/fleurs", "data/en_us/test.tsv", repo_type="dataset"), en_path)
    en = _tsv_transcriptions(en_path)
    ids = sorted(set(ta) & set(en))
    pairs = [(i, ta[i], en[i]) for i in ids]
    return pairs[:limit] if limit else pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=None, help="override translate.model")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    import sacrebleu

    pairs = load_pairs(args.n)
    tr = IndicTranslator(model_id=args.model)
    if not tr.available():
        print(f"model not cached: {tr.model_id}  -> python scripts/download_indictrans.py", file=sys.stderr)
        return 1
    print(f"==> {tr.model_id} ({tr.backend})  on {len(pairs)} FLEURS/FLoRes ta->en pairs")

    hyps, refs = [], []
    t0 = time.perf_counter()
    for k, (sid, ta, en) in enumerate(pairs, 1):
        out = tr.translate(ta, force=True)
        hyps.append(out.target if out else "")
        refs.append(en)
        if k % 25 == 0:
            print(f"    [{k}/{len(pairs)}]", flush=True)
    elapsed = time.perf_counter() - t0

    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    print("\n" + "=" * 70)
    print(f"TRANSLATION ta->en  [{tr.model_id} on {tr.resolved_device}]")
    print("=" * 70)
    print(f"  sentences : {len(pairs)}")
    print(f"  chrF      : {chrf.score:.1f}")
    print(f"  BLEU      : {bleu.score:.1f}")
    print(f"  speed     : {elapsed / len(pairs):.2f} s/sentence")
    for sid, ta, en in pairs[: args.show]:
        h = hyps[pairs.index((sid, ta, en))]
        print(f"\n  --- {sid} ---\n  ta : {ta[:100]}\n  ref: {en[:100]}\n  hyp: {h[:100]}")

    out = ARTIFACT_DIR / "reports" / "translation_eval.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"results": {}}
    prev["results"][tr.model_id] = {
        "backend": tr.backend, "n": len(pairs), "chrf": chrf.score, "bleu": bleu.score,
        "sec_per_sentence": elapsed / len(pairs), "device": tr.resolved_device,
        "samples": [{"id": s, "ta": t, "ref": r, "hyp": h} for (s, t, r), h in list(zip(pairs, hyps))[:5]],
    }
    prev["gold"] = "FLEURS ta_in/test x en_us/test joined on FLoRes sentence id"
    out.write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
