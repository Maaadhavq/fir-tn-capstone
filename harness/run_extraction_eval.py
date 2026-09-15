"""Per-slot precision / recall for an extractor against a gold set.

    python -m harness.run_extraction_eval                      # rules extractor
    python -m harness.run_extraction_eval --gold path.jsonl    # another gold set

Gold format (one JSON object per line):

    {"id": "...", "text": "...", "gold": {"date": [...iso...], "time": [...HH:MM...],
                                          "amount_inr": [...], "phone": [...], "vehicle": [...]}}

Scoring is on **normalised values**, set-based per (document, slot): a date
counts as correct if the extractor produced the same ISO string, regardless of
where in the text it found it. Offsets are checked separately in the unit
tests; this harness answers "did we get the right facts", which is what the
form cares about.

The 15-example gold set under tests/fixtures/ was authored by hand and is
deliberately adversarial (impossible dates, bare numbers, account numbers that
look like phones, two-phone cases). It is a smoke set, not a benchmark -- the
real Stage B evaluation needs the synthetic corpus.

Any extractor exposing `extract_all(text) -> dict[Kind, list[Span]]` can be
plugged in, which is how the NER / LLM extractors will be compared later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import ARTIFACT_DIR  # noqa: E402
from fir.extract.rules import Kind, extract_all  # noqa: E402

DEFAULT_GOLD = REPO_ROOT / "tests" / "fixtures" / "extraction_gold.jsonl"
SLOTS = [k.value for k in Kind]


def _norm(v) -> str:
    """Compare amounts numerically, everything else as strings."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def evaluate(gold_path: Path, extractor=extract_all) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    per_doc: list[dict] = []

    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        found = extractor(ex["text"])
        doc = {"id": ex["id"], "errors": []}
        for slot in SLOTS:
            g = {_norm(v) for v in ex["gold"].get(slot, [])}
            p = {_norm(s.value) for s in found.get(Kind(slot), [])}
            tp[slot] += len(g & p)
            fp[slot] += len(p - g)
            fn[slot] += len(g - p)
            if p - g:
                doc["errors"].append(f"{slot} FP {sorted(p - g)}")
            if g - p:
                doc["errors"].append(f"{slot} FN {sorted(g - p)}")
        per_doc.append(doc)

    def prf(t, f_p, f_n):
        pr = t / (t + f_p) if (t + f_p) else 1.0
        rc = t / (t + f_n) if (t + f_n) else 1.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        return pr, rc, f1

    rows = {}
    for slot in SLOTS:
        pr, rc, f1 = prf(tp[slot], fp[slot], fn[slot])
        rows[slot] = {"tp": tp[slot], "fp": fp[slot], "fn": fn[slot],
                      "precision": pr, "recall": rc, "f1": f1}
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    pr, rc, f1 = prf(T, FP, FN)
    rows["_micro"] = {"tp": T, "fp": FP, "fn": FN, "precision": pr, "recall": rc, "f1": f1}
    return {"gold": str(gold_path), "n_docs": len(per_doc), "slots": rows, "per_doc": per_doc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--name", default="rules")
    args = ap.parse_args()

    rep = evaluate(args.gold)
    print(f"extractor: {args.name}   gold: {args.gold.name}   docs: {rep['n_docs']}\n")
    print(f"{'slot':<12s} {'tp':>4s} {'fp':>4s} {'fn':>4s}   {'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 50)
    for slot, r in rep["slots"].items():
        label = "micro" if slot == "_micro" else slot
        if slot == "_micro":
            print("-" * 50)
        print(f"{label:<12s} {r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d}   "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}")

    errs = [d for d in rep["per_doc"] if d["errors"]]
    if errs:
        print(f"\n{len(errs)} document(s) with errors:")
        for d in errs:
            for e in d["errors"]:
                print(f"  {d['id']}: {e}")

    out = ARTIFACT_DIR / "reports" / f"extraction_{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
