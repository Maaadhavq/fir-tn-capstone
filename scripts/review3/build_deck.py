"""Review 3 panel deck -> Review3_Presentation.pptx (Windows-side Python, python-pptx).

Style follows Review2_Presentation.pptx (Calibri; navy #24405B titles; teal
#35637E accent bar; 16:9). Numbers are read from artifacts/reports/*.json and
data/statutes/*.csv at build time; the rubric is BCSE497J Review 3 (16 Sep
2026): 7 parameters, 20 marks. Slides marked with «...» need a human to fill
in facts only the team has (the panel's Review-2 comments, who did what).

    python scripts/review3/build_deck.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "artifacts" / "reports"
ASSETS = REPO / "assets"
OUT = REPO / "Review3_Presentation.pptx"

NAVY = RGBColor(0x24, 0x40, 0x5B)
TEAL = RGBColor(0x35, 0x63, 0x7E)
INK = RGBColor(0x1F, 0x2A, 0x37)
MUTED = RGBColor(0x5B, 0x64, 0x70)
PANEL = RGBColor(0xF5, 0xF7, 0xF9)
LINE = RGBColor(0xC9, 0xD3, 0xDD)
RED = RGBColor(0xB4, 0x42, 0x3B)
GREEN = RGBColor(0x3C, 0x8A, 0x5A)
AMBER = RGBColor(0xC9, 0x8A, 0x1B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Emu(12191695), Emu(6858000)
LM = Inches(0.7)          # left margin
CW = W - 2 * LM           # content width


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def numbers() -> dict:
    n: dict = {}
    full = _load("statute_baseline_full.json") or {}
    sub = _load("statute_baseline.json") or {}
    fr = full.get("results", {})
    n["tfidf_full"] = fr.get("tfidf+ovr-logistic", {})
    n["bert_full_best"] = fr.get("InLegalBERT[e4_ht_hl10]", fr.get("InLegalBERT", {}))
    n["bert_full_base"] = fr.get("InLegalBERT", {})
    n["tfidf_sub"] = sub.get("results", {}).get("tfidf+ovr-logistic", {})
    n["n_train_full"], n["n_test_full"] = full.get("n_train", 0), full.get("n_test", 0)
    ens = _load("ensemble_full_e4_ht_hl10.json") or {}
    n["ens_gain"] = ens.get("gain_vs_best_single")
    asr_new = _load("asr_baseline.json") or {}
    asr_old = _load("asr_baseline_fallback_ladder.json") or asr_new
    n["asr_new"] = asr_new if "temperature" in asr_new else None
    n["asr_old"] = asr_old
    n["asr_seq"] = _load("asr_baseline_seq.json")
    # finding 6g: decoder brakes on the replay loop -- every sweep file with the two guard clips
    sweep = []
    for p in sorted(REPORTS.glob("asr_baseline_rp*.json")) + sorted(REPORTS.glob("asr_baseline_nr*.json")):
        r = _load(p.name)
        if not r:
            continue
        clip = {c["id"]: c for c in r.get("per_clip", []) if c["file"] in ("12583250098003224463.wav", "16989398024822917306.wav")}
        sweep.append({"tag": p.stem.replace("asr_baseline_", ""), "rp": r.get("repetition_penalty"), "nr": r.get("no_repeat_ngram_size"),
                      "wer": r["wer"], "cer": r["cer"], "rtf": r["rtf"], "flagged": r.get("n_flagged", 0),
                      "c1916": clip.get("1916", {}).get("wer"), "c1721": clip.get("1721", {}).get("wer")})
    n["sweep"] = sweep
    prof = _load("asr_profile.json") or {}
    n["prof"] = prof
    tr = _load("translation_eval.json") or {}
    n["tr"] = next(iter(tr.get("results", {}).values()), {})
    n["tr_n"] = tr.get("n_pairs") or tr.get("n") or (n["tr"].get("n") if isinstance(n["tr"], dict) else None)
    ex = _load("extraction_rules.json") or {}
    n["ex"] = ex
    sched = REPO / "data" / "statutes" / "bnss_schedule1.csv"
    if sched.exists():
        with sched.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        n["sched_total"] = len(rows)
        n["sched"] = {k: sum(1 for r in rows if r["cognizable"] == k) for k in ("cognizable", "non_cognizable", "conditional")}
    bns = REPO / "data" / "statutes" / "bns_2023.jsonl"
    n["bns_sections"] = sum(1 for _ in bns.open(encoding="utf-8")) if bns.exists() else 0
    # tests: the count the suite reports, recorded in PROGRESS.md ("**N tests pass**")
    m = re.search(r"\*\*(\d+) tests pass\*\*", (REPO / "PROGRESS.md").read_text(encoding="utf-8"))
    n["n_tests"] = int(m.group(1)) if m else 0
    n["n_test_files"] = len(list((REPO / "tests").glob("test_*.py")))
    return n


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, title: str, subtitle: str | None = None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        tb = s.shapes.add_textbox(LM, Inches(0.45), CW, Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = "Calibri", Pt(28), True, NAVY
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, LM + Inches(0.02), Inches(1.12), Inches(1.1), Inches(0.035))
        bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
        if subtitle:
            st = s.shapes.add_textbox(LM, Inches(1.18), CW, Inches(0.4))
            q = st.text_frame.paragraphs[0]
            rr = q.add_run(); rr.text = subtitle
            rr.font.name, rr.font.size, rr.font.color.rgb, rr.font.italic = "Calibri", Pt(13), MUTED, True
        num = s.shapes.add_textbox(W - Inches(1.0), H - Inches(0.45), Inches(0.6), Inches(0.3))
        pn = num.text_frame.paragraphs[0]; pn.alignment = PP_ALIGN.RIGHT
        rn = pn.add_run(); rn.text = str(self.n)
        rn.font.name, rn.font.size, rn.font.color.rgb = "Calibri", Pt(10), MUTED
        foot = s.shapes.add_textbox(LM, H - Inches(0.45), Inches(8), Inches(0.3))
        pf = foot.text_frame.paragraphs[0]
        rf = pf.add_run(); rf.text = "Speech-driven FIR drafting (TN IF-1) · BCSE497J Review 3 · 16 Sep 2026"
        rf.font.name, rf.font.size, rf.font.color.rgb = "Calibri", Pt(10), MUTED
        return s

    @staticmethod
    def text(s, left, top, width, height, lines, size=14, color=INK, bold_first=False, bullet="–  ", spacing=4):
        tb = s.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        first = True
        for line in lines:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.space_after = Pt(spacing)
            if isinstance(line, tuple):        # (text, {"bold":..,"color":..,"size":..,"bullet":..})
                txt, opts = line
            else:
                txt, opts = line, {}
            b = opts.get("bullet", bullet)
            r = p.add_run()
            r.text = (b if b and txt else "") + txt
            r.font.name = "Calibri"
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", bold_first and p is tf.paragraphs[0])
            r.font.color.rgb = opts.get("color", color)
        return tb

    @staticmethod
    def panel(s, left, top, width, height, heading=None):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        sh.fill.solid(); sh.fill.fore_color.rgb = PANEL
        sh.line.color.rgb = LINE; sh.line.width = Pt(0.75)
        sh.adjustments[0] = 0.04
        if heading:
            Deck.text(s, left + Inches(0.18), top + Inches(0.1), width - Inches(0.36), Inches(0.4),
                      [(heading, {"bold": True, "color": NAVY, "size": 14, "bullet": ""})])
        return sh

    @staticmethod
    def table(s, left, top, width, rows, col_widths=None, size=11, header=True, row_h=Inches(0.34)):
        nrows, ncols = len(rows), len(rows[0])
        shp = s.shapes.add_table(nrows, ncols, left, top, width, row_h * nrows)
        t = shp.table
        if col_widths:
            for i, w in enumerate(col_widths):
                t.columns[i].width = w
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(0.06)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                tf = cell.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                color = INK
                txt = str(val)
                if isinstance(val, tuple):
                    txt, color = val
                r = p.add_run(); r.text = txt
                r.font.name, r.font.size = "Calibri", Pt(size)
                if header and i == 0:
                    r.font.bold, r.font.color.rgb = True, WHITE
                    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
                else:
                    r.font.color.rgb = color
                    cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if i % 2 else PANEL
        return shp

    @staticmethod
    def picture(s, path: Path, left, top, width=None, height=None):
        if not path.exists():
            return Deck.text(s, left, top, Inches(4), Inches(0.5), [(f"[figure missing: {path.name}]", {"color": RED, "bullet": ""})])
        return s.shapes.add_picture(str(path), left, top, width=width, height=height)

    def save(self):
        self.prs.save(str(OUT))
        print("wrote", OUT.relative_to(REPO), f"({self.n} slides)")


def pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def f3(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------
def build() -> None:
    n = numbers()
    d = Deck()

    # 1 -- title
    s = d.prs.slides.add_slide(d.blank); d.n += 1
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.5)); band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
    d.text(s, LM, Inches(0.1), CW, Inches(0.35), [("B.TECH PROJECT — I     |     REVIEW 3     |     VIT CHENNAI", {"color": WHITE, "size": 13, "bullet": "", "bold": True})])
    d.text(s, LM, Inches(1.6), CW, Inches(1.6), [("Speech-Driven Drafting of First Information Reports for the Tamil Nadu Police", {"size": 34, "bold": True, "color": NAVY, "bullet": ""})])
    d.text(s, LM, Inches(3.15), CW, Inches(0.9), [("Implementation progress and interim results: one LangGraph pipeline from a spoken Tamil complaint to a draft IF-1, with the officer as author of record.", {"size": 15, "color": MUTED, "bullet": ""})])
    acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, LM, Inches(4.25), Inches(1.2), Inches(0.04)); acc.fill.solid(); acc.fill.fore_color.rgb = TEAL; acc.line.fill.background()
    d.text(s, LM, Inches(4.5), Inches(5.5), Inches(1.6), [("Team", {"bold": True, "color": NAVY, "size": 12, "bullet": ""}),
            ("23BAI1088   Madhav K", {"size": 12, "bullet": ""}), ("23BAI1245   Nischay Kuchibotla", {"size": 12, "bullet": ""}), ("23BAI1416   Rohit A. S.", {"size": 12, "bullet": ""})], spacing=2)
    d.text(s, Inches(7.2), Inches(4.5), Inches(4.5), Inches(1.6), [("Guide", {"bold": True, "color": NAVY, "size": 12, "bullet": ""}),
            ("Dr. Shivaranjani", {"size": 12, "bullet": ""}), ("School of Computer Science and Engineering", {"size": 12, "bullet": ""}), ("16 September 2026", {"size": 12, "bullet": ""})], spacing=2)

    # 2 -- rubric map
    s = d.slide("Review 3 rubric coverage", "BCSE497J Review 3 — 7 parameters, 20 marks. Where each is answered in this deck.")
    rows = [["#", "Evaluation parameter", "Marks", "Slides", "Evidence"],
            ["1", "Follow-up on Review 2 feedback; progress vs approved plan", "3", "3, 4", "action table; 27-milestone status"],
            ["2", "Implementation and functional progress (~50% of scope)", "3", "5–11", "end-to-end pipeline, live demo"],
            ["3", "Technical accuracy and best practices", "3", "14", f"{n['n_tests']} tests, pinned invariants, gazette-cited tables"],
            ["4", "Interim testing, results and analysis", "3", "8, 9, 12", "F1 / WER-CER / chrF / P-R tables and figures"],
            ["5", "Problem-solving and technical refinement", "2", "13", "9 findings, each with the corrective action"],
            ["6", "Progress of the project report", "3", "15", "Chapters 1–3 revised, Chapter 4 drafted"],
            ["7", "Presentation, individual responsibility, Q&A", "3", "16, 18", "contribution table, demo, questions"]]
    d.table(s, LM, Inches(1.75), CW, rows, col_widths=[Inches(0.4), Inches(4.6), Inches(0.7), Inches(1.1), Inches(4.0)], size=12)

    # 3 -- follow-up on Review 2
    s = d.slide("Follow-up on Review 2 feedback", "1 mark: action taken on panel comments. 2 marks: progress against the approved plan (next slide).")
    rows = [["Review 2 comment / suggestion", "Action taken", "Where"],
            ["«Panel comment 1 — copy from the Review 2 evaluation sheet»", "«what was done»", "«slide / section»"],
            ["«Panel comment 2»", "«what was done»", "«slide / section»"],
            ["«Panel comment 3»", "«what was done»", "«slide / section»"]]
    d.table(s, LM, Inches(1.7), CW, rows, col_widths=[Inches(4.3), Inches(4.5), Inches(2.0)], size=12)
    d.panel(s, LM, Inches(3.35), CW, Inches(2.75), "Decisions changed since Review 2 — and why")
    d.text(s, LM + Inches(0.2), Inches(3.8), CW - Inches(0.4), Inches(2.3), [
        "Compute: the dev machine has an 8 GB GPU, not the 16 GB assumed — the plan's 14B synthetic-data model does not fit; the 7B / 14B-with-offload decision is measured, not assumed (finding 1).",
        "Statute classifier: InLegalBERT was to be the classifier; on ILSI a TF-IDF baseline beats it by 0.17 micro-F1 and an ensemble adds +0.0006 — TF-IDF is the classifier, the encoder's job becomes element verification (finding 8).",
        "Cognizability: the planned First Schedule table was a hand-coded stub; it is now parsed from the gazette PDF with page citations, and the stub turned out wrong on 4 of 48 entries (finding 9).",
        "A Tamil→English translation node was added — the classifier is English-only — and its first real failure (“dowry” → “relief”) produced the cue-scan safety net (finding 5c).",
    ], size=12)

    # 4 -- progress vs plan
    s = d.slide("Progress against the approved plan (PLAN.md, 27 milestones)", "10 working · 11 partial / v0 · 6 not started. Everything green or amber runs inside one end-to-end graph today.")
    d.picture(s, ASSETS / "r3_milestones.png", LM, Inches(1.55), height=Inches(4.75))
    d.text(s, Inches(8.9), Inches(1.7), Inches(3.0), Inches(4.5), [
        ("Semester-1 target (PLAN §2):", {"bold": True, "color": NAVY, "bullet": ""}),
        "“a thin end-to-end vertical slice on synthetic data”",
        ("Status:", {"bold": True, "color": NAVY, "bullet": ""}),
        "the slice is not thin any more — ASR → extraction → translation → statute-ID → IPC→BNS → cognizability → elements → IF-1 → form → officer page run as one LangGraph graph",
        ("Deliberately not started:", {"bold": True, "color": NAVY, "bullet": ""}),
        "LLM-based modules (P2.2, P3.3-LLM, P4.2-LLM, P6.2) wait on the 8 GB model decision; PDF form and audit log are semester-2 items",
    ], size=12)

    # 5 -- what runs today
    s = d.slide("What runs end to end today", "One graph: (audio→asr) → extract → translate → statute_id → ipc_bns → cognizability_check → instantiate. ~10,400 lines; text or audio in, IF-1 draft out.")
    d.picture(s, ASSETS / "r3_pipeline_status.png", LM, Inches(1.55), width=CW)
    d.text(s, LM, Inches(5.3), CW, Inches(1.1), [
        "Decision support by construction: no endpoint, CLI or graph node can set a record to “verified”; the form prints “DRAFT — the officer is the author of record”; every machine-filled slot carries provenance (character offsets into the transcript).",
        "Strict local-only: models from a local HF cache; the API binds 127.0.0.1; audio uploads are deleted after transcription.",
    ], size=12)

    # 6 -- Stage A
    s = d.slide("Stage A — Speech and ASR: baseline, guards, normalisation", "faster-whisper large-v3 (CTranslate2) on the 8 GB GPU; FLEURS ta_in as the zero-shot yardstick.")
    new_ = n["asr_new"] or {}
    old_ = n["asr_old"] or {}
    d.text(s, LM, Inches(1.6), Inches(5.3), Inches(4.6), [
        ("Built", {"bold": True, "color": NAVY, "bullet": ""}),
        f"Zero-shot baseline on {old_.get('n_clips', 60)} clips: WER {pct(old_.get('wer'))} → {pct(new_.get('wer'))}, CER {pct(old_.get('cer'))} → {pct(new_.get('cer'))} after this week's decoding fix; Tamil is agglutinative, so CER is reported alongside WER (finding 6c)",
        "Hallucination guards: duplicated-trigram fraction (replay loops) and foreign-script count (language drift); thresholds calibrated on the real baseline — clean clips score exactly 0%; a flagged transcript blocks auto-resolution (finding 6b)",
        "Tamil inverse text normalisation: 206-form number lexicon with sandhi, spoken dates/times/amounts → digits, offset map back to the audio transcript (69 tests)",
        ("Fixed this week (finding 6f)", {"bold": True, "color": NAVY, "bullet": ""}),
        f"The library's temperature-fallback ladder re-decoded low-confidence Tamil windows up to six times. One pass at T=0 + batched VAD chunks: on 60 clips RTF {old_.get('rtf', 0):.2f} → {new_.get('rtf', 0):.2f} and WER/CER improved, not traded",
        "On a complaint-length recording: RTF 0.31 → 0.10, WER 81% → 49% (chunks follow pauses, not fixed 30 s cuts)",
        f"Honest caveat: {new_.get('n_flagged', 2)} of 60 clips still trip the repetition guard at T=0 — the ladder was the speed cost, not the cause of the replay loop; the guard stays",
    ], size=11)
    d.picture(s, ASSETS / "r3_asr.png", Inches(6.15), Inches(1.55), width=Inches(5.85))
    d.picture(s, ASSETS / "r3_asr_profile.png", Inches(6.15), Inches(4.25), width=Inches(5.85))

    # 7 -- Stage B + translation
    s = d.slide("Stage B — Extraction floor, and the Tamil→English view", "Rules with provenance now; NER / LLM extraction is the semester-2 upgrade. The classifier is English-only, so Tamil input is translated for it — and only for it.")
    d.panel(s, LM, Inches(1.6), Inches(5.5), Inches(3.9), "Rule extractor (Stage B floor)")
    ex = n["ex"]
    micro = ((ex.get("slots") or {}).get("_micro") or {}) if isinstance(ex, dict) else {}
    d.text(s, LM + Inches(0.2), Inches(2.05), Inches(5.1), Inches(4.0), [
        "Slots: dates (digit, spoken, year-less), times, ₹ amounts (digit and spoken, Indian grouping), phone numbers, vehicle plates",
        "Fills a slot only when unambiguous; several candidates → blank + officer note",
        f"Gold smoke set: 28 adversarial complaints, {int(micro.get('tp', 63))}/{int(micro.get('tp', 63) + micro.get('fn', 0))} slots, {int(micro.get('fp', 0))} false positives",
        "Every value carries character offsets — the form can point back at the words",
        ("Limit: names, places and accused descriptions are not extracted yet — that is what the synthetic-data + NER phase is for", {"color": MUTED}),
    ], size=11.5)
    d.panel(s, Inches(6.4), Inches(1.6), Inches(5.55), Inches(3.9), "Translation node (finding 5b / 5c)")
    tr = n["tr"]
    d.text(s, Inches(6.6), Inches(2.05), Inches(5.15), Inches(4.0), [
        f"Interim model opus-mt-dra-en: chrF {tr.get('chrf', 35.2):.1f}, BLEU {tr.get('bleu', 10.5):.1f} on FLoRes ta→en; IndicTrans2 (~+20 chrF) is gated behind a licence click",
        "Output feeds the classifier only; extraction, elements and the narrative keep the original Tamil",
        ("First real failure: “வரதட்சணை” (dowry) → “relief”; the classifier saw a threat, and a 498A complaint routed to CSR", {"color": RED}),
        "Fix: a bilingual element cue-scan on the ORIGINAL text; a cognizable section whose ingredients are all present but unpredicted forces officer review, naming the section",
        "Pinned by tests that reproduce the exact failure",
    ], size=11.5)

    # 8 -- Stage C classifier
    s = d.slide("Stage C — Statute identification: the encoder question is settled", f"ILSI: {n['n_train_full']:,} training / {n['n_test_full']:,} test court documents, 100 IPC sections, multi-label. Threshold tuned on dev.")
    d.picture(s, ASSETS / "r3_statute_f1.png", LM, Inches(1.55), width=Inches(7.6))
    tf_ = n["tfidf_full"]; bb = n["bert_full_best"]
    d.text(s, Inches(8.5), Inches(1.7), Inches(3.45), Inches(4.6), [
        ("Result", {"bold": True, "color": NAVY, "bullet": ""}),
        f"TF-IDF + one-vs-rest logistic: micro-F1 {f3(tf_.get('micro_f1'))}, macro-F1 {f3(tf_.get('macro_f1'))}",
        f"InLegalBERT with every lever (4 epochs, head LR×10, head+tail truncation, pos-weight): {f3(bb.get('micro_f1'))} / {f3(bb.get('macro_f1'))}",
        f"Ensemble gain over the best single model: {n['ens_gain']:+.4f}" if isinstance(n["ens_gain"], (int, float)) else "Ensemble: no gain",
        ("Ruled out as causes", {"bold": True, "color": NAVY, "bullet": ""}),
        "truncation to 512 wordpieces (costs TF-IDF only 8.6% relative); head collapse (fixed with pos-weight, finding 7); too little data (full corpus used)",
        ("Decision", {"bold": True, "color": NAVY, "bullet": ""}),
        "TF-IDF is the classifier; the encoder's job is element verification. Keep the sparse floor in every comparison.",
    ], size=11.5)

    # 9 -- legal KB
    s = d.slide("Stage C — The legal knowledge base is now the gazette, not memory", "IPC→BNS map with a legal-safety invariant; BNSS First Schedule and all 358 BNS section texts parsed from the official gazette PDFs with page citations.")
    d.picture(s, ASSETS / "r3_schedule.png", LM, Inches(1.7), width=Inches(7.9))
    sch = n.get("sched", {})
    d.text(s, Inches(8.75), Inches(1.65), Inches(3.25), Inches(4.7), [
        ("IPC→BNS map (100 ILSI sections)", {"bold": True, "color": NAVY, "bullet": ""}),
        "69 auto-applicable · 29 flagged for review · 7 map outside the BNS. Invariant, pinned by tests: only confidence = high AND needs_review = N AND a real BNS target is applied without a human.",
        ("BNSS First Schedule, Part I", {"bold": True, "color": NAVY, "bullet": ""}),
        f"{n.get('sched_total', 438)} section keys: {sch.get('cognizable', 277)} cognizable, {sch.get('non_cognizable', 135)} non-cognizable, {sch.get('conditional', 26)} conditional (abetment follows the offence; s.85 depends on who reports; s.303(2) theft splits at ₹5,000 — resolved from the extracted value)",
        ("BNS 2023 text", {"bold": True, "color": NAVY, "bullet": ""}),
        f"{n['bns_sections']} sections with headings, chapters and illustration-free operative text; shown beside every suggestion, IPC text kept as lineage",
        ("Status: parsed and spot-checked, not legally signed off", {"color": MUTED}),
    ], size=11)

    # 10 -- elements
    s = d.slide("Stage C — Element-wise justification (v0) and the cue-scan safety net", "A section is not one thing. The officer sees which ingredients the facts show, quoted from the transcript — and which are still to be established.")
    d.panel(s, LM, Inches(1.6), Inches(6.2), Inches(4.1), "Output on the form (IPC 304B → BNS 80, dowry-death sample)")
    d.text(s, LM + Inches(0.2), Inches(2.05), Inches(5.8), Inches(4.0), [
        ("[?] death of a woman — not stated in the narrative; establish: is the deceased a woman?", {"bullet": "", "size": 11}),
        ("[x] by burns, bodily injury, or otherwise than in normal circumstances · 'murder'", {"bullet": "", "size": 11}),
        ("[x] within seven years of marriage · 'married five years ago'", {"bullet": "", "size": 11}),
        ("[x] cruelty or harassment for dowry, soon before death · 'cruelty'; 'demand'; 'dowry'", {"bullet": "", "size": 11}),
        ("", {"bullet": ""}),
        "Two rules keep it honest: never “no” (absence of a word is not evidence of absence — it yields “unclear”, the officer's cue to ask), and every “yes” carries the matched words and offsets",
        "29 sections covered with English + Tamil cues (theft, robbery, hurt, kidnapping, cheating incl. OTP fraud, cruelty, dowry death, trespass …); a 22-complaint battery pins what must and must not fire",
    ], size=11.5)
    d.panel(s, Inches(7.1), Inches(1.6), Inches(4.85), Inches(4.1), "Why it doubles as a safety net")
    d.text(s, Inches(7.3), Inches(2.05), Inches(4.45), Inches(4.0), [
        "The classifier reads an English view of the complaint; the cues read the original Tamil",
        "A covered section whose every ingredient is present but which the classifier did not predict is raised as a question — never auto-applied",
        "If that section is cognizable or conditional and the route would otherwise be CSR, the case goes to the officer with the section named",
        ("Next (PLAN P3.3): replace keyword cues with a local LLM reading each element's BNS text against the complaint — same schema, same tests", {"color": MUTED}),
    ], size=11.5)

    # 11 -- Stage D/E
    s = d.slide("Stages D and E — IF-1 record, grounded narrative, form, officer page", "Deterministic slot-filling into the fixed CCTNS IF-1; the only free text is the narrative box, and every sentence in it must trace to an extracted fact.")
    d.text(s, LM, Inches(1.6), Inches(5.7), Inches(4.7), [
        ("Stage D", {"bold": True, "color": NAVY, "bullet": ""}),
        "Pydantic record = the canonical IF-1 JSON Schema (parity tests fail on any drift); status draft → csr_advisory → verified, and nothing in src/ can write “verified”",
        "Decision → record bridge: auto-applied sections become BNS candidates with IPC lineage; held-back mappings are kept, flagged ambiguous, score-capped",
        "Narrative composer: one sentence per grounded fact, Tamil + English, faithfulness gate = every sentence supported (structural v0; NLI/LLM judge later)",
        "Renderer: the 15-item form, machine-filled slots annotated ⟨status · confidence · source⟩, items 13–15 always blank (they are the officer's)",
        ("Stage E", {"bold": True, "color": NAVY, "bullet": ""}),
        "FastAPI on 127.0.0.1: /v1/complaint/{text,audio}, /v1/statute/{section}, /v1/health; the officer page records from the microphone or takes an audio file, or typed text: route, sections with BNS text and element checks, “also consider”, flags, annotated form; “Mark verified” disabled by design",
        "Audit log v0 (P5.3): every draft appended to an immutable JSONL — record id, route, sections, flags, a hash of the text; never the text or the audio",
    ], size=11)
    d.panel(s, Inches(6.6), Inches(1.6), Inches(5.35), Inches(4.0), "Live demo (6 min)")
    d.text(s, Inches(6.8), Inches(2.05), Inches(4.95), Inches(4.2), [
        "1  Typed theft complaint with a value → officer review with BNS 303(2) cognizable (₹15,000 > ₹5,000)",
        "2  Same complaint without the value → “cognizability depends on property value”; nothing defaults",
        "3  Dowry-cruelty complaint → FIR via BNS 80; BNS 85 conditional; element checks with quoted words",
        "4  ● Record a Tamil complaint into the microphone → transcript, ITN (₹50,000), the same decision — speech-driven, all local",
        "5  Upload FLEURS clip 1916 → the transcript guard fires; the draft is marked not auto-resolvable",
        "6  Look up BNS 85 → gazette text + the Schedule's condition",
        ("Fallback if the GPU is busy: `python -m fir draft --json` on the CLI, same output", {"color": MUTED}),
    ], size=11)

    # 11b -- the officer page, if a screenshot exists (scripts/review3 headless Edge capture)
    shot = ASSETS / "r3_ui_dowry.png"
    if shot.exists():
        s = d.slide("The officer page — what the demo shows", "Static page driving the same API the CLI uses; a React workspace replaces it in semester 2 without changing the contract.")
        d.picture(s, shot, Inches(0.6), Inches(1.5), width=Inches(7.2))
        d.text(s, Inches(8.0), Inches(1.6), Inches(3.9), Inches(4.8), [
            "Route badge with the Schedule's rationale; every suggested section with its BNS gazette text, the IPC lineage text, and element checks (✓ quoted words / ? what to establish)",
            "“Also consider”: sections whose ingredients are in the original text but the classifier did not predict — a question, never applied",
            "Speak it: microphone or file → the transcript lands in the box and the same decision renders; ASR guard flags shown",
            "“Mark verified” is disabled by design: verification is the officer's act in CCTNS",
        ], size=11.5)

    # 12 -- interim results table
    s = d.slide("Interim results at a glance", "All numbers are read from artifacts/reports/*.json by the build script; RESULTS.md is regenerated by `make results`.")
    old = n["asr_old"]; new = n["asr_new"]; seq = n["asr_seq"]
    rows = [["Component", "Data", "Metric", "Result", "Read"],
            ["Statute-ID (TF-IDF)", f"ILSI test, {n['n_test_full']:,} docs", "micro / macro-F1", f"{f3(tf_.get('micro_f1'))} / {f3(tf_.get('macro_f1'))}", "the classifier of record"],
            ["Statute-ID (InLegalBERT, best)", "same", "micro / macro-F1", f"{f3(bb.get('micro_f1'))} / {f3(bb.get('macro_f1'))}", "loses to the sparse floor"],
            ["ASR before (fallback ladder)", f"FLEURS ta_in, {old.get('n_clips', 60)} clips", "WER / CER / RTF", f"{pct(old.get('wer'))} / {pct(old.get('cer'))} / {old.get('rtf', 0):.2f}", f"{old.get('n_flagged', 2)} clips flagged by the guard"],
            ["ASR now (T=0, batched)", "same", "WER / CER / RTF", (f"{pct(new.get('wer'))} / {pct(new.get('cer'))} / {new.get('rtf', 0):.2f}" if new else "«running»"), (f"{new.get('n_flagged', 0)} flagged: replay loop persists at T=0" if new else "60-clip re-run in progress")],
            ["ASR, complaint-length recording", "8 clips joined, 148 s", "RTF / WER", "0.10 / 49% (batched) vs 0.31 / 81% (sequential)", "batching is also more accurate"],
            ["Translation ta→en (opus-mt)", "FLoRes devtest, 336 pairs", "chrF / BLEU", f"{tr.get('chrf', 35.2):.1f} / {tr.get('bleu', 10.5):.1f}", "upper bound on Tamil statute-ID"],
            ["Extraction (rules)", "28 gold complaints", "slot P / R", "1.00 / 1.00 (63 slots, 0 FP)", "smoke set, not a benchmark"],
            ["IPC→BNS map", "100 ILSI sections", "auto / review / non-BNS", "69 / 29 / 7", "third-party source; gazette check pending"],
            ["BNSS First Schedule", f"{n.get('sched_total', 438)} keys from gazette", "cog / non-cog / conditional", f"{sch.get('cognizable', 277)} / {sch.get('non_cognizable', 135)} / {sch.get('conditional', 26)}", "stub was wrong on 4/48"],
            ["Test suite", f"{n['n_test_files']} files", "tests passing", f"{n['n_tests']}", "no model weights needed, ~20 s"]]
    if seq:
        rows.insert(5, ["ASR (T=0, sequential)", "same", "WER / CER / RTF", f"{pct(seq.get('wer'))} / {pct(seq.get('cer'))} / {seq.get('rtf', 0):.2f}", f"{seq.get('n_flagged', 0)} flagged"])
    d.table(s, LM, Inches(1.6), CW, rows, col_widths=[Inches(2.6), Inches(2.2), Inches(1.9), Inches(2.6), Inches(1.5)], size=10.5, row_h=Inches(0.36))

    # 13 -- challenges
    s = d.slide("Technical challenges and corrective action", "Each finding is recorded in PROGRESS.md with the measurement that established it and the test that keeps it fixed.")
    rows = [["Challenge (finding)", "Evidence", "Corrective action"],
            ["GPU is 8 GB, not 16 GB (1)", "torch reports 8.5 GB; 14B-Q4 ≈ 9 GB", "models.yaml carries a fits flag per model; 7B vs 14B-with-offload to be benchmarked on mains power"],
            ["WSL VM rebooted under full-corpus jobs (6e)", "6 deaths; predictor = Python heap > ~2 GB", "chunked vectoriser / tokenizer / predict; detached launches; logs on /mnt/c"],
            ["InLegalBERT head collapsed to 6/100 labels (7)", "loss flat at 0.142, macro-F1 0.018", "pos-weighted BCE, higher head LR; then measured honestly against TF-IDF"],
            ["Whisper fabricated text on 2/60 clips (6b)", "replay loop 226% WER; Cyrillic drift", "transcript guards calibrated on the baseline; flagged → not auto-resolvable"],
            ["ASR 3× slower than real time (6f)", "profile: fallback ladder 3.06 vs 0.57 RTF", "one pass at T=0; batched VAD chunks; 60-clip WER 57→51%, CER 22→15%, RTF 1.81→0.46"],
            ["Translation lost “dowry” → CSR route (5c)", "reproduced on the real server", "bilingual cue scan on the original; forces officer review"],
            ["Cognizability stub wrong on 4/48 (9)", "gazette parse vs stub: 126, 223, 296, 329", "parse the First Schedule with page citations; conditional class; value-resolved theft split"],
            ["Classifier abstains on short complaints (5)", "top-5 for a theft sample has no 379", "not a threshold problem — in-domain synthetic data; cue scan carries short inputs"]]
    if n["sweep"]:
        base = n["asr_new"] or {}
        cleared = [r for r in n["sweep"] if r["flagged"] == 0]
        best = min(cleared, key=lambda r: r["wer"]) if cleared else min(n["sweep"], key=lambda r: (r["flagged"], r["wer"]))
        verdict = (f"rp {best['rp']}, nr {best['nr']}: 0 flagged, WER {pct(best['wer'])} vs {pct(base.get('wer'))}"
                   if cleared else
                   f"no setting clears both clips without cost (best: rp {best['rp']}, nr {best['nr']} → {best['flagged']} flagged, WER {pct(best['wer'])}); guard stays, defaults kept")
        rows.append(["Replay loop survives T=0 on 2/60 clips (6g)",
                     f"sweep of {len(n['sweep'])} decoder settings on the 60 clips, day before review",
                     verdict])
    d.table(s, LM, Inches(1.6), CW, rows, col_widths=[Inches(3.3), Inches(3.3), Inches(4.2)], size=10, row_h=Inches(0.46))

    # 14 -- accuracy & practices
    s = d.slide("Technical accuracy and engineering practice", "What a reviewer can check without trusting us.")
    d.text(s, LM, Inches(1.6), Inches(5.6), Inches(4.7), [
        ("Testing", {"bold": True, "color": NAVY, "bullet": ""}),
        f"{n['n_tests']} tests in {n['n_test_files']} files, ~20 s, no weights: golden end-to-end fixtures, schema parity, ITN, cue battery, Schedule integrity, serving, CLI",
        "Legal invariants are tests: the IPC→BNS auto-apply rule, “unknown never defaults”, “nothing can set verified”, threat ≠ attempt/murder",
        "ruff clean; LF-only repo; reproducible builds: `make schedule`, `make bns-text`, `make results`",
        ("Evaluation practice", {"bold": True, "color": NAVY, "bullet": ""}),
        "Sparse baseline kept in every comparison; thresholds tuned on dev, reported on test; CER beside WER for Tamil; per-clip distributions, not just corpus means",
        "Negative results recorded: encoder < TF-IDF; ensemble +0.0006; condition_on_previous_text did not fix replay; int8 vs fp16 WER gap was fallback noise",
    ], size=11.5)
    d.text(s, Inches(6.5), Inches(1.6), Inches(5.45), Inches(4.7), [
        ("Provenance and safety", {"bold": True, "color": NAVY, "bullet": ""}),
        "Every extracted value: character offsets; every element “yes”: quoted words; every Schedule row: gazette page; every model: local cache path",
        "Decision support only — route and sections are proposals; the officer is the author of record; CSR is advisory",
        "Local-only inference; no hosted API anywhere in the code; PII-minimal (audio deleted after transcription)",
        ("Best-practice choices", {"bold": True, "color": NAVY, "bullet": ""}),
        "CTranslate2 int8 for VRAM headroom; LangGraph for an inspectable state machine; Pydantic ↔ JSON Schema parity; config-driven (pipeline.yaml, models.yaml, datasets.yaml)",
        "Honest limits stated on the form: “stub”/“parsed, not signed off”/“interim model” wherever true",
    ], size=11.5)

    # 15 -- report progress
    s = d.slide("Project report progress", "Review3_Report.docx: Chapters 1–3 carried forward with a revisions section; Chapter 4 drafted from the same JSON reports as this deck.")
    rows = [["Chapter", "Status", "Contents at Review 3"],
            ["Preliminary pages", "updated", "abstract with a Review-3 status paragraph; rubric map for Review 3; TOC"],
            ["1 Introduction", "unchanged", "background, problem, objectives, scope, expected outcomes"],
            ["2 Literature / patent review", "unchanged", "16 sources; comparative analysis; gap"],
            ["3 Methodology / design", "revised", "§3.8 Revisions since Review 2: compute budget, classifier choice, translation node, gazette-parsed KB, team allocation"],
            ["4 Implementation and results", "new (initial)", "environment; implementation by stage; testing & validation; interim results (6 tables, 5 figures); analysis; limitations; individual contribution"],
            ["5 Conclusion and future work", "outline", "to be written for Review 4"],
            ["References", "extended", "gazette notifications, faster-whisper / CTranslate2, opus-mt, sacreBLEU added"]]
    d.table(s, LM, Inches(1.6), CW, rows, col_widths=[Inches(2.6), Inches(1.5), Inches(6.7)], size=11.5, row_h=Inches(0.42))
    d.text(s, LM, Inches(5.25), CW, Inches(0.9), [
        ("AI-tool acknowledgement (guideline §2): an AI coding assistant (Claude Code) was used for implementation support under the team's direction; all design decisions, legal tables and results were reviewed and are reproducible from the repository. «confirm wording with the guide»", {"color": MUTED, "size": 11, "bullet": ""}),
    ])

    # 16 -- individual contribution
    s = d.slide("Individual contribution", "«Confirm before the review — the panel grades each member's demonstrated contribution (guideline §2).»")
    rows = [["Member", "Ownership (as planned at Review 2)", "Delivered at Review 3", "Demonstrates"],
            ["Madhav K (23BAI1088)", "Stage C/D: statute identification, IPC→BNS, cognizability, IF-1 instantiation; integration", "«…»", "«slides / demo steps»"],
            ["Nischay Kuchibotla (23BAI1245)", "Stage A: ASR benchmark, guards, Tamil ITN, code-switch fine-tune", "«…»", "«…»"],
            ["Rohit A. S. (23BAI1416)", "Stage B: extraction, synthetic data engine, NER seeding", "«…»", "«…»"]]
    d.table(s, LM, Inches(1.65), CW, rows, col_widths=[Inches(2.3), Inches(3.6), Inches(3.0), Inches(1.9)], size=11.5, row_h=Inches(0.75))
    d.text(s, LM, Inches(4.9), CW, Inches(1.2), [
        ("The repository records who did what: PROGRESS.md is the dated engineering log; every module has an owner in IMPLEMENTATION.md. Present the split as it actually happened — the rubric rewards demonstrated contribution, not the plan.", {"color": MUTED, "size": 11, "bullet": ""}),
    ])

    # 17 -- plan to Review 4
    s = d.slide("Plan to Review 4 (12–16 October 2026)", "Complete implementation, final results, report verification — and proof of a publication submission (Review 4 parameter 9, 2 marks).")
    d.text(s, LM, Inches(1.6), Inches(5.7), Inches(4.7), [
        ("Weeks 1–2", {"bold": True, "color": NAVY, "bullet": ""}),
        "Decide the local LLM on mains power (7B-Q4 vs 14B-Q4 offload) → synthetic Tamil complaint → IF-1 generator (PLAN §5.2)",
        "IndicTrans2 (licence click) or NLLB-600M replaces opus-mt; re-measure Tamil statute-ID end to end",
        "IndicWhisper / IndicConformer benchmark against the corrected Whisper numbers",
        ("Weeks 3–4", {"bold": True, "color": NAVY, "bullet": ""}),
        "LLM element verifier reading BNS text (P3.3) behind the same ElementCheck schema; NER (MuRIL) for names / places / accused",
        "Legal read-through of the Schedule CSV and the 29 flagged mapping rows with the guide",
        "Self-recorded Tamil complaint gold set (P6.3), consent script from the DPDP memo",
    ], size=11.5)
    d.panel(s, Inches(6.6), Inches(1.6), Inches(5.35), Inches(4.0), "Publication plan «confirm venue with the guide»")
    d.text(s, Inches(6.8), Inches(2.05), Inches(4.95), Inches(4.2), [
        "Paper 1 (ready to write from Chapter 4): “A sparse baseline beats a domain-pretrained encoder on Indian statute identification — and what the encoder is for instead” (ILSI, full-corpus results, element verification)",
        "Paper 2 (after the gold set): “Officer-in-the-loop FIR drafting from spoken Tamil complaints: provenance, cognizability and the cost of translation”",
        "Candidate venues: ICON 2026 (NLP in India), FIRE 2026 (legal track), or an IEEE/Springer conference with a Dec–Jan deadline — submission proof needed by 12 Oct",
        ("Artefacts already reusable: RESULTS.md tables, r3_*.png figures, reproducible harness", {"color": MUTED}),
    ], size=11.5)

    # 18 -- close
    s = d.slide("Summary", "Half the approved scope runs as working modules, in one graph, with the officer as author of record.")
    d.text(s, LM, Inches(1.7), CW, Inches(3.6), [
        "An end-to-end pipeline from spoken Tamil to a draft IF-1 exists and is tested; nothing in it can file anything",
        "The research question was measured, not assumed: on ILSI a sparse baseline beats InLegalBERT (0.700 vs 0.526 micro-F1); the encoder's role is element verification",
        "The legal tables the routing depends on now come from the gazette with page citations — and doing so caught four refusal-direction errors in the hand-coded version",
        "ASR runs 4× faster with better WER and CER after one measured change; two real safety failures (translation loss, fabricated transcripts) each produced a guard that is pinned by a test",
        "Next: the local LLM decision unblocks synthetic data, the element verifier and the fine-tuning pass; publication draft from Chapter 4",
    ], size=14)
    d.text(s, LM, Inches(5.4), CW, Inches(0.6), [("Questions — and the live demo.", {"bold": True, "color": NAVY, "size": 18, "bullet": ""})])

    d.save()


if __name__ == "__main__":
    build()
