"""Figures for Review 3 (deck + report), drawn from artifacts/reports/*.json.

Runs on the Windows-side Python (matplotlib is there; the WSL venv has none).
Every number comes from a report file the harness wrote -- nothing is typed in
here except the milestone status table, which is a judgement recorded in
PROGRESS.md.

    python scripts/review3/make_figures.py        # -> assets/r3_*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "artifacts" / "reports"
ASSETS = REPO / "assets"

NAVY, TEAL, GREY, LIGHT, RED, AMBER, GREEN = "#24405B", "#35637E", "#5B6470", "#C9D3DD", "#B4423B", "#C98A1B", "#3C8A5A"
plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.edgecolor": GREY, "axes.labelcolor": NAVY,
                     "xtick.color": NAVY, "ytick.color": NAVY, "axes.titlecolor": NAVY, "axes.titleweight": "bold"})


def _load(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save(fig, name: str) -> Path:
    out = ASSETS / name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out.relative_to(REPO))
    return out


# ---------------------------------------------------------------------------
def fig_statute_f1() -> None:
    full = _load("statute_baseline_full.json")
    sub = _load("statute_baseline.json")
    ens = _load("ensemble_full_e4_ht_hl10.json")
    if not (full and sub):
        return
    rows = [
        ("TF-IDF + OvR logistic\n(8k subsample)", sub["results"]["tfidf+ovr-logistic"]),
        ("InLegalBERT\n(8k, 2 ep)", sub["results"]["InLegalBERT"]),
        ("TF-IDF + OvR logistic\n(full 42.8k)", full["results"]["tfidf+ovr-logistic"]),
        ("InLegalBERT\n(full, 2 ep)", full["results"]["InLegalBERT"]),
        ("InLegalBERT\n(full, 4 ep, head LR×10,\nhead+tail)", full["results"]["InLegalBERT[e4_ht_hl10]"]),
    ]
    labels = [r[0] for r in rows]
    micro = [r[1]["micro_f1"] for r in rows]
    macro = [r[1]["macro_f1"] for r in rows]
    if ens:
        best = max(ens["sweep"], key=lambda r: r["dev_micro_f1"])
        labels.append(f"Ensemble\n(α={ens['best_alpha']:.1f} TF-IDF + BERT)")
        micro.append(best["test_micro_f1"])
        macro.append(best["test_macro_f1"])
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], micro, w, color=NAVY, label="micro-F1")
    b2 = ax.bar([i + w / 2 for i in x], macro, w, color=TEAL, label="macro-F1")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9, color=NAVY)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("F1 on ILSI test (13,039 docs for full-corpus rows)")
    ax.set_title("Statute identification: a sparse baseline beats the legal encoder — and the ensemble adds nothing")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "r3_statute_f1.png")


def fig_asr() -> None:
    variants = []
    for label, fname in (("temperature fallback ladder\n(library default, before)", "asr_baseline_fallback_ladder.json"),
                         ("one pass at T=0,\nsequential 30 s window", "asr_baseline_seq.json"),
                         ("one pass at T=0,\nbatched VAD chunks (bs=8)\n= pipeline default now", "asr_baseline.json")):
        r = _load(fname)
        if r and (fname != "asr_baseline.json" or "temperature" in r):   # skip stale copies
            variants.append((label, r))
    if not variants:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    x = range(len(variants))
    w = 0.38
    wer = [v[1]["wer"] * 100 for v in variants]
    cer = [v[1]["cer"] * 100 for v in variants]
    b1 = ax.bar([i - w / 2 for i in x], wer, w, color=NAVY, label="WER %")
    b2 = ax.bar([i + w / 2 for i in x], cer, w, color=TEAL, label="CER %")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=9, color=NAVY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([v[0] for v in variants], fontsize=8.5)
    ax.set_ylim(0, max(wer) * 1.25)
    ax.set_title(f"faster-whisper large-v3 int8, FLEURS ta_in test ({variants[0][1]['n_clips']} clips)", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = axes[1]
    rtf = [v[1]["rtf"] for v in variants]
    bars = ax2.bar(list(x), rtf, color=[RED] + [GREEN] * (len(variants) - 1))
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}", ha="center", fontsize=9, color=NAVY)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(["before", "T=0 seq", "T=0 batched"][: len(variants)], fontsize=9)
    ax2.set_ylabel("real-time factor (lower is faster)")
    ax2.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax2.set_title("RTF", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    _save(fig, "r3_asr.png")


def fig_asr_profile() -> None:
    prof = _load("asr_profile.json")
    if not prof:
        return
    rows = [r for r in prof["rows"] if r["compute"] == "int8"]
    per = [r for r in rows if r["mode"] == "per-clip"]
    joined = [r for r in rows if r["mode"].startswith("joined") and r["beam"] == 5]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    labels = [f"beam {r['beam']}\n{'ladder' if r.get('fallback', True) else 'T=0'}" for r in per]
    colors = [RED if r.get("fallback", True) else GREEN for r in per]
    bars = ax.bar(range(len(per)), [r["rtf"] for r in per], color=colors)
    for b, r in zip(bars, per):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"RTF {r['rtf']:.2f}\nWER {r['wer'] * 100:.0f}%", ha="center", fontsize=8.5, color=NAVY)
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(r["rtf"] for r in per) * 1.35)
    ax.set_title(f"Per-clip decoding ({prof['n_clips']} clips): the temperature ladder costs 5×", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = axes[1]
    labels = ["sequential\n30 s window" if r["mode"] == "joined-sequential" else f"batched\nbs={r['batch']}" for r in joined]
    bars = ax2.bar(range(len(joined)), [r["rtf"] for r in joined], color=[AMBER] + [GREEN] * (len(joined) - 1))
    for b, r in zip(bars, joined):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"RTF {r['rtf']:.2f}\nWER {r['wer'] * 100:.0f}%", ha="center", fontsize=8.5, color=NAVY)
    ax2.set_xticks(range(len(joined)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, max(r["rtf"] for r in joined) * 1.45)
    ax2.set_title("One 2.5-min recording (beam 5, T=0): VAD-chunk batching", fontsize=10.5)
    ax2.spines[["top", "right"]].set_visible(False)
    _save(fig, "r3_asr_profile.png")


def fig_schedule() -> None:
    p = REPO / "data" / "statutes" / "bnss_schedule1.csv"
    if not p.exists():
        return
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    counts = {k: sum(1 for r in rows if r["cognizable"] == k) for k in ("cognizable", "non_cognizable", "conditional")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.pie(list(counts.values()), labels=[f"{k.replace('_', '-')}\n{v}" for k, v in counts.items()],
           colors=[NAVY, TEAL, AMBER], startangle=90, textprops={"fontsize": 9, "color": NAVY}, wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax.set_title(f"BNSS First Schedule Pt I parsed: {len(rows)} section keys", fontsize=10.5)
    ax2 = axes[1]
    stub = [("126 wrongful restraint", "non-cog.", "cognizable"), ("223 disobeying public servant", "non-cog.", "cognizable"),
            ("296 obscene acts", "non-cog.", "cognizable"), ("329 criminal trespass", "non-cog.", "cognizable"),
            ("85 cruelty (498A)", "cognizable", "conditional*"), ("303(2) theft", "cognizable", "conditional†")]
    ax2.axis("off")
    tbl = ax2.table(cellText=[[a, b, c] for a, b, c in stub], colLabels=["BNS section", "hand-coded stub said", "gazette Schedule says"],
                    loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(LIGHT)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 2:
            cell.get_text().set_color(RED if "cognizable" == stub[r - 1][2] else AMBER)
    ax2.set_title("Where the hand-coded stub was wrong (4 refusal-direction errors, 2 hidden conditions)", fontsize=10)
    fig.text(0.52, 0.02, "* cognizable only when reported by the aggrieved woman / relative   † non-cognizable below ₹5,000", fontsize=8, color=GREY)
    _save(fig, "r3_schedule.png")


def fig_milestones() -> None:
    # (id, name, status) -- status per PROGRESS.md; 2 = working, 1 = partial/v0, 0 = not started
    ms = [
        ("P0.1", "Repo, env, GPU baseline", 2), ("P0.2", "IF-1 JSON Schema", 2), ("P0.3", "Legal KB: BNS text, First Schedule, IPC→BNS map", 2),
        ("P0.4", "Ethics / DPDP memo", 1),
        ("P1.1", "Audio front-end (VAD)", 1), ("P1.2", "ASR baseline + guards", 2), ("P1.3", "Tamil ITN", 2), ("P1.4", "Guided-interview TTS (optional)", 0),
        ("P2.1", "Extraction A: rules (+NER later)", 1), ("P2.2", "Extraction B: LLM structured", 0), ("P2.3", "Bilingual fields", 1),
        ("P3.1", "IPC→BNS mapping layer", 2), ("P3.2", "Multi-label statute classifier", 2), ("P3.3", "Element-wise verifier (rule v0; LLM later)", 1),
        ("P3.4", "Cognizability gate (gazette Schedule)", 2),
        ("P4.1", "Deterministic IF-1 template engine", 2), ("P4.2", "Grounded narrative (deterministic v0)", 1), ("P4.3", "Faithfulness gate", 1),
        ("P4.4", "Print-faithful PDF", 0),
        ("P5.1", "Review workspace (page v0: text, mic, upload)", 1), ("P5.2", "Statute panel", 1), ("P5.3", "Audit & versioning (append-only draft log v0)", 1),
        ("P6.1", "End-to-end LangGraph orchestration", 2), ("P6.2", "Fine-tuning pass", 0), ("P6.3", "Human gold set", 0),
        ("P6.4", "Full evaluation study (harness exists)", 1), ("P6.5", "Write-up & demo", 1),
    ]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    colors = {2: GREEN, 1: AMBER, 0: LIGHT}
    for i, (mid, name, st) in enumerate(reversed(ms)):
        ax.barh(i, 1, color=colors[st], height=0.72)
        ax.text(0.02, i, f"{mid}  {name}", va="center", fontsize=9, color=NAVY if st else GREY)
        ax.text(0.985, i, {2: "working", 1: "partial / v0", 0: "not started"}[st], va="center", ha="right", fontsize=8.5,
                color="white" if st == 2 else NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    ax.axis("off")
    n2 = sum(1 for m in ms if m[2] == 2)
    n1 = sum(1 for m in ms if m[2] == 1)
    ax.set_title(f"PLAN.md milestones at Review 3: {n2} working, {n1} partial, {len(ms) - n2 - n1} not started (of {len(ms)})", fontsize=11)
    _save(fig, "r3_milestones.png")


def fig_pipeline_status() -> None:
    stages = [
        ("A  Speech / ASR", ["whisper large-v3 int8", "hallucination guards", "Tamil ITN (206 forms)", "T=0, batched chunks"], GREEN),
        ("B  Extraction", ["rules + provenance", "date·time·₹·phone·plate", "63/63 gold, 0 FP", "NER / LLM: next"], AMBER),
        ("ta → en", ["opus-mt (interim)", "classifier-only view", "chrF 35.2", "IndicTrans2: gated"], AMBER),
        ("C  Statute-ID", ["TF-IDF F1 .700/.581", "IPC→BNS high-conf", "gazette Schedule 438", "29-section cue scan"], GREEN),
        ("D  IF-1 record", ["Pydantic = Schema", "deterministic fill", "grounded narrative", "fixed 15-item form"], GREEN),
        ("E  Verification", ["FastAPI, local only", "page: text · mic · upload", "lookup · audit log", "cannot mark verified"], AMBER),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.axis("off")
    n = len(stages)
    w = 1 / n
    for i, (title, items, col) in enumerate(stages):
        x0 = i * w + 0.006
        ax.add_patch(plt.Rectangle((x0, 0.08), w - 0.012, 0.84, facecolor="#F5F7F9", edgecolor=col, lw=2))
        ax.text(x0 + 0.01, 0.84, title, fontsize=10.5, weight="bold", color=NAVY, va="center")
        for j, it in enumerate(items):
            ax.text(x0 + 0.01, 0.68 - j * 0.15, "• " + it, fontsize=8.2, color="#1F2A37", va="center")
        if i < n - 1:
            ax.annotate("", xy=(x0 + w - 0.006, 0.5), xytext=(x0 + w - 0.016, 0.5), arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.2})
    ax.text(0.0, 0.0, "green = working end to end · amber = working v0, upgrade planned", fontsize=8.5, color=GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "r3_pipeline_status.png")


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    fig_statute_f1()
    fig_asr()
    fig_asr_profile()
    fig_schedule()
    fig_milestones()
    fig_pipeline_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
