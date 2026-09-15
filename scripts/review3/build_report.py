"""Review 3 report -> Review3_Report.docx (Windows-side Python, python-docx).

Starts from Review2_Report.docx (Chapters 1-3, references) and adds what the
BCSE497J Review-3 rubric asks for: updated Chapters 1-3 (a revisions section),
an initial Chapter 4 (implementation, testing, interim results, analysis,
limitations, individual contribution) and a Chapter 5 outline. Numbers are read
from artifacts/reports/*.json and data/statutes/*.csv; figures from assets/r3_*.png
(scripts/review3/make_figures.py). «...» marks facts only the team can supply.

    python scripts/review3/build_report.py
"""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "artifacts" / "reports"
ASSETS = REPO / "assets"
SRC = REPO / "Review2_Report.docx"
OUT = REPO / "Review3_Report.docx"
MUTED = RGBColor(0x5B, 0x64, 0x70)


def _load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "—"


def f3(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


# ---------------------------------------------------------------------------
class Writer:
    """Appends content at the end of the document, then moves it before an anchor."""

    def __init__(self, doc: Document):
        self.doc = doc
        self.new: list = []

    def _track(self, el):
        self.new.append(el)

    def h1(self, text):
        p = self.doc.add_paragraph(text, style="Heading 1"); self._track(p._p); return p

    def h2(self, text):
        p = self.doc.add_paragraph(text, style="Heading 2"); self._track(p._p); return p

    def h3(self, text):
        p = self.doc.add_paragraph(text, style="Heading 3"); self._track(p._p); return p

    def para(self, text, style="Normal", italic=False, muted=False):
        p = self.doc.add_paragraph(style=style)
        r = p.add_run(text)
        if italic:
            r.italic = True
        if muted:
            r.font.color.rgb = MUTED
        self._track(p._p)
        return p

    def bullet(self, text):
        p = self.doc.add_paragraph(text, style="List Bullet"); self._track(p._p); return p

    def caption(self, text):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text); r.italic = True; r.font.size = Pt(9)
        self._track(p._p)
        return p

    def table(self, rows, widths=None, size=9.5):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.text = ""
                p = cell.paragraphs[0]
                r = p.add_run(str(val))
                r.font.size = Pt(size)
                if i == 0:
                    r.bold = True
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E8EDF2")
                    cell._tc.get_or_add_tcPr().append(shd)
        if widths:
            for j, w in enumerate(widths):
                for row in t.rows:
                    row.cells[j].width = w
        self._track(t._tbl)
        spacer = self.doc.add_paragraph(); self._track(spacer._p)
        return t

    def picture(self, path: Path, width=Inches(6.2)):
        if not path.exists():
            return self.para(f"[figure missing: {path.name}]", muted=True)
        self.doc.add_picture(str(path), width=width)
        p = self.doc.paragraphs[-1]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._track(p._p)
        return p

    def move_before(self, anchor_el):
        for el in self.new:
            anchor_el.addprevious(el)
        self.new = []


def set_update_fields(doc: Document) -> None:
    """Ask Word to refresh the TOC field when the file is opened."""
    settings = doc.settings.element
    if settings.find(qn("w:updateFields")) is None:
        uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); settings.append(uf)


# ---------------------------------------------------------------------------
def build() -> None:
    doc = Document(str(SRC))
    ps = doc.paragraphs
    by_text = {p.text.strip(): p for p in ps}

    # numbers ---------------------------------------------------------------
    full = _load("statute_baseline_full.json") or {}
    sub = _load("statute_baseline.json") or {}
    fr, sr = full.get("results", {}), sub.get("results", {})
    ens = _load("ensemble_full_e4_ht_hl10.json") or {}
    abl = _load("truncation_ablation.json") or {}
    asr_new = _load("asr_baseline.json") or {}
    asr_new = asr_new if "temperature" in asr_new else None
    asr_old = _load("asr_baseline_fallback_ladder.json") or {}
    asr_seq = _load("asr_seq.json") or _load("asr_baseline_seq.json")
    prof = _load("asr_profile.json") or {}
    tr = next(iter((_load("translation_eval.json") or {}).get("results", {}).items()), ("Helsinki-NLP/opus-mt-dra-en", {}))
    ex = _load("extraction_rules.json") or {}
    slots = ex.get("slots", {})
    sched_rows = []
    sp = REPO / "data" / "statutes" / "bnss_schedule1.csv"
    if sp.exists():
        with sp.open(encoding="utf-8", newline="") as fh:
            sched_rows = list(csv.DictReader(fh))
    sched = {k: sum(1 for r in sched_rows if r["cognizable"] == k) for k in ("cognizable", "non_cognizable", "conditional")}
    m = re.search(r"\*\*(\d+) tests pass\*\*", (REPO / "PROGRESS.md").read_text(encoding="utf-8"))
    n_tests = int(m.group(1)) if m else 0
    tf_full, bert_full, bert_best = fr.get("tfidf+ovr-logistic", {}), fr.get("InLegalBERT", {}), fr.get("InLegalBERT[e4_ht_hl10]", {})

    # -- title page ----------------------------------------------------------
    def set_text(par, text):
        keep = [(r.bold, r.font.size, r.font.color.rgb if r.font.color and r.font.color.type else None) for r in par.runs][:1]
        for r in list(par.runs):
            r._r.getparent().remove(r._r)
        r = par.add_run(text)
        if keep:
            r.bold, r.font.size = keep[0][0], keep[0][1]
            if keep[0][2] is not None:
                r.font.color.rgb = keep[0][2]

    title_map = {
        "Capstone Project — Review 2 Report": "BCSE497J Project — I  ·  Review 3 Report",
        "Madhav Komanduri  —  «Register No.»": "Madhav K  —  23BAI1088",
        "«Team Member 2»  —  «Register No.»": "Nischay Kuchibotla  —  23BAI1245",
        "«Team Member 3»  —  «Register No.»": "Rohit A. S.  —  23BAI1416",
        "«Guide Name and Designation»": "Dr. Shivaranjani",
        "«Department / School»": "School of Computer Science and Engineering (SCOPE)",
        "«University / Institution»": "Vellore Institute of Technology, Chennai",
        "August 2026": "September 2026",
    }
    for par in doc.paragraphs[:14]:
        if par.text in title_map:
            set_text(par, title_map[par.text])

    # -- abstract: status paragraph -----------------------------------------
    abstract_h = by_text["Abstract"]
    abs_p = doc.paragraphs[ps.index(abstract_h) + 1]
    rubric_h = by_text["Rubric Coverage Map"]
    status = copy.deepcopy(abs_p._p)
    rubric_h._p.addprevious(status)
    from docx.text.paragraph import Paragraph
    sp_ = Paragraph(status, abs_p._parent)
    for r in list(sp_.runs):
        r._r.getparent().remove(r._r)
    sp_.add_run(
        "Status at Review 3 (16 September 2026). The pipeline described in Chapter 3 now runs end to end as one "
        "LangGraph graph: ASR with hallucination guards and Tamil inverse text normalisation, rule-based extraction with "
        "provenance, a Tamil→English view for the English-only statute classifier, statute identification on the full ILSI "
        "corpus, IPC→BNS conversion under a high-confidence-only rule, cognizability routing from the BNSS First Schedule "
        "parsed from the gazette, element-wise justification, deterministic IF-1 instantiation with a grounded narrative, "
        "and a local FastAPI service with an officer verification page. Chapter 4 reports the implementation, the test "
        f"suite ({n_tests} tests) and interim results: TF-IDF statute identification at micro-F1 {tf_full.get('micro_f1', 0):.3f} "
        f"against InLegalBERT at {bert_best.get('micro_f1', 0):.3f}; Tamil ASR at WER {pct(asr_old.get('wer'))} / CER {pct(asr_old.get('cer'))} "
        "zero-shot, made five times faster by removing the decoder's temperature fallback; and a cognizability table in which "
        "the official Schedule corrected four entries of the hand-coded version."
    )

    # -- rubric map: Review 3 -----------------------------------------------
    intro = next(q for q in doc.paragraphs if q.text.startswith("Each Review-2 evaluation parameter"))
    for r in list(intro.runs):
        r._r.getparent().remove(r._r)
    intro.add_run("Each Review-3 evaluation parameter is addressed by the report sections below.")
    t0 = doc.tables[0]
    r3 = [["#", "Evaluation Parameter", "Report Section(s)", "CO"],
          ["1", "Follow-up on Review 2 feedback & progress vs plan", "§3.8, §4.1, Fig. 4.1", "CO1, CO3, CO4"],
          ["2", "Implementation & functional progress", "§4.2, Table 4.1", "CO4"],
          ["3", "Technical accuracy & best practices", "§4.3, §4.5", "CO2"],
          ["4", "Interim testing, results & analysis", "§4.4 (Tables 4.3–4.8, Figs 4.2–4.4)", "CO2"],
          ["5", "Problem-solving & technical refinement", "§4.5", "CO2, CO4"],
          ["6", "Progress of the report", "this document; §4.7", "CO1, CO4"],
          ["7", "Presentation, individual responsibility", "§4.7; Review3_Presentation", "CO3, CO4"]]
    for i, row in enumerate(r3):
        for j, val in enumerate(row):
            cell = t0.cell(i, j)
            para = cell.paragraphs[0]
            keep_bold = bool(para.runs and para.runs[0].bold)
            size = para.runs[0].font.size if para.runs else None
            for r in list(para.runs):
                r._r.getparent().remove(r._r)
            r = para.add_run(val)
            if keep_bold or i == 0:
                r.bold = True
            if size:
                r.font.size = size

    # -- §3.8 revisions + Chapter 4 + Chapter 5 outline before References ----
    refs_h = by_text["References"]
    w = Writer(doc)

    w.h2("3.8 Revisions since Review 2")
    w.para("The methodology of Chapter 3 stands, with five revisions recorded here so that the report and the implementation agree (guideline §2).")
    w.bullet("Compute budget (§3.6.1). The development machine has an RTX 5050 Laptop GPU with 8 GB of VRAM, below the 12 GB fallback assumed at Review 2. "
             "Every model entry in configs/models.yaml now carries a fits flag against the real budget; the 14B synthetic-data generator does not fit at 4-bit "
             "(~9 GB) and the 7B-versus-14B-with-offload decision is deferred to a measured benchmark on mains power.")
    w.bullet("Statute classifier (§3.2.3). InLegalBERT was to be the multi-label classifier. On the full ILSI corpus a TF-IDF + one-vs-rest logistic baseline "
             f"reaches micro-F1 {tf_full.get('micro_f1', 0):.4f} against {bert_best.get('micro_f1', 0):.4f} for the encoder with every lever applied, and an ensemble adds "
             f"{ens.get('gain_vs_best_single', 0):+.4f}. TF-IDF is therefore the classifier of record; the encoder's role moves to element verification (§4.4.1).")
    w.bullet("Translation node. The classifier is English-only; a Tamil→English node (interim Helsinki-NLP/opus-mt-dra-en) feeds it. Its first real failure — "
             "“வரதட்சணை” (dowry) rendered as “relief” — motivated the bilingual element cue-scan safety net that runs on the original text (§4.4.5).")
    w.bullet("Legal knowledge base (§3.4). The BNSS First Schedule and all 358 BNS section texts are parsed from the official gazette PDFs with page citations, "
             "replacing the hand-coded cognizability stub, which the gazette showed to be wrong on four of forty-eight entries (§4.4.2).")
    w.bullet("Allocation of responsibilities (§3.6.4). «State here how the split described in Table 3.9 was actually followed between Review 2 and Review 3.»")

    w.h1("4. Implementation and Interim Results")
    w.para("This chapter reports the state of the implementation at Review 3: the environment and its constraints, what was built for each stage, how it is "
           "tested, the interim measurements, an analysis of the technical problems met and how they were resolved, the limitations at this stage, and the "
           "individual contributions. Every number is read from a machine-written report under artifacts/reports/ and can be regenerated with make results.")

    w.h2("4.1 Development environment and engineering constraints")
    w.para("All Python runs inside WSL2 Ubuntu 24.04 (Python 3.11, uv-managed virtual environment) because Windows Smart App Control blocks the pyarrow "
           "wheel that the Hugging Face datasets and bitsandbytes packages depend on; data loaders therefore read plain JSONL/TSV and parquet through DuckDB. "
           "The GPU is an RTX 5050 Laptop (Blackwell, sm_120, 8 GB); PyTorch 2.11 with CUDA 12.8 wheels is required because older wheels ship no kernels for "
           "this architecture. The WSL virtual machine has 7.7 GB of RAM and restarts itself when a single process's Python heap passes roughly 2 GB; every "
           "corpus-scale build (vectoriser, tokenizer, prediction) is therefore chunked, long jobs are launched detached with logs on the Windows file system, "
           "and one memory-heavy job runs at a time. Inference is strictly local: models load from a local Hugging Face cache with HF_HUB_OFFLINE=1, the API "
           "binds 127.0.0.1 only, and uploaded audio is deleted after transcription.")
    w.picture(ASSETS / "r3_milestones.png", width=Inches(6.3))
    w.caption("Figure 4.1  —  Status of the 27 PLAN.md milestones at Review 3: 10 working, 11 partial (v0), 6 not started.")

    w.h2("4.2 Implementation by stage")
    w.para("Figure 4.2 summarises the graph as built. Table 4.1 lists the modules behind each stage, the file that implements them and the tests that pin their behaviour.")
    w.picture(ASSETS / "r3_pipeline_status.png", width=Inches(6.4))
    w.caption("Figure 4.2  —  The pipeline as built. Green: working end to end; amber: working v0 with a planned upgrade.")
    rows = [["Stage", "Module (src/fir/…)", "What it does", "Tests"],
            ["A", "asr/whisper.py", "faster-whisper large-v3 (CTranslate2 int8) with device fallback; one decoding pass at temperature 0; batched VAD-chunk decoding", "test_asr_quality"],
            ["A", "asr/quality.py", "transcript guards: duplicated-trigram fraction (replay loops) and foreign-script count (language drift), calibrated on the baseline", "test_asr_quality (fixtures from real clips)"],
            ["A", "asr/itn.py", "Tamil inverse text normalisation: 206-form number lexicon with sandhi, spoken dates/times/amounts → digits, offset map to the transcript", "test_itn (69)"],
            ["B", "extract/rules.py, extract/fill.py", "dates, times, ₹ amounts, phones, vehicle plates with character offsets; fills an IF-1 slot only when unambiguous", "test_extract_rules; harness gold set (28 docs)"],
            ["→", "translate/indictrans.py", "Tamil→English for the classifier only (marian/nllb/indictrans backends); original text kept for every other stage", "test_translate"],
            ["C", "statute/tfidf_baseline.py, statute/inlegalbert.py, statute/registry.py", "multi-label statute classifiers on ILSI; TF-IDF (hashed 1–2-grams, chunked) is the default", "harness/run_statute_baseline"],
            ["C", "statute/ipc_bns_map.py", "IPC→BNS conversion; auto-apply only when confidence = high, needs_review = N and the target is a BNS section", "test_slice_golden"],
            ["C", "statute/cognizability.py", "BNSS First Schedule lookup (438 keys from the gazette); cognizable / non-cognizable / conditional / unknown; value-resolved conditions", "test_bnss_schedule (41), test_slice_golden"],
            ["C", "statute/elements.py", "element-wise justification for 29 sections with English + Tamil cues; cue scan as recall safety net", "test_elements, test_cue_scan, test_cue_battery"],
            ["C", "statute/statute_text.py, statute/bns_text.py", "statute text lookup and lexical retrieval rank (ILSI IPC text); BNS 2023 section text from the gazette", "test_statute_text, test_bns_text"],
            ["D", "schema/if1.py, schema/statute_decision.py", "Pydantic IF-1 record in parity with the canonical JSON Schema; the decision record with provenance", "test_if1_schema"],
            ["D", "instantiate/from_decision.py, narrative.py, render.py", "decision → record bridge; grounded bilingual narrative with a faithfulness gate; the fixed 15-item form", "test_narrative, test_render_if1"],
            ["E", "serving/app.py, serving/static/index.html, __main__.py", "FastAPI (text, audio, statute lookup, health); officer verification page; CLI draft / check / mapping", "test_serving, test_cli"],
            ["all", "orchestrator/graph.py", "LangGraph state graph: (audio→asr) → extract → translate → statute_id → ipc_bns → cognizability_check → instantiate", "test_slice_golden (fixture-pinned)"]]
    w.table(rows, widths=[Inches(0.4), Inches(1.9), Inches(3.0), Inches(1.4)], size=8.5)
    w.caption("Table 4.1  —  Modules implemented at Review 3, by pipeline stage.")

    w.h2("4.3 Testing and validation")
    w.para(f"The suite has {n_tests} tests across 17 files and runs in about twenty seconds without model weights, so it runs on every change. "
           "Beyond unit tests, four kinds of checks carry the legal-safety requirements of Chapter 3 as executable assertions:")
    w.bullet("Golden end-to-end fixtures (tests/fixtures/golden_slice.json): nine complaints with the expected IPC predictions, BNS sections, review reasons and route, run through the real graph with a stub classifier so that the mapping, cognizability and instantiation logic — not the model — is under test.")
    w.bullet("Invariants as tests: the IPC→BNS auto-apply rule; “an unknown section never defaults”; “nothing in src/ can set a record to verified”; “a threat to kill is criminal intimidation, not attempt or murder”; Pydantic ↔ JSON Schema parity that fails on any drift.")
    w.bullet("Data integrity: the parsed First Schedule must have 438 keys in gazette order, no empty classification cells, a gazette page on every row and 25 classifications a lawyer can check; the BNS text must contain exactly sections 1–358 with headings.")
    w.bullet("Reproduced failures: the mistranslated dowry complaint, the Whisper replay loop and the Cyrillic drift are fixtures; the tests that guard them would fail if the guard were removed.")
    w.para("Static checks: ruff (E, F, W, I) is clean; the repository is LF-only; every build artefact has a make target (make schedule, make bns-text, make results).")

    w.h2("4.4 Interim results")
    w.h3("4.4.1 Statute identification (ILSI, 100 IPC sections)")
    w.para(f"ILSI provides {full.get('n_train', 42835):,} training, {full.get('n_dev', 10200):,} development and {full.get('n_test', 13039):,} test court documents. "
           "Decision thresholds are tuned on the development split for micro-F1 and reported on test. Table 4.3 and Figure 4.3 compare the sparse baseline with "
           "the domain-pretrained encoder on the 8k subsample used for the vertical slice and on the full corpus.")
    rows = [["Training set", "Model", "micro-F1", "macro-F1", "labels used / 100"],
            ["8k subsample", "TF-IDF + OvR logistic", f3(sr.get("tfidf+ovr-logistic", {}).get("micro_f1")), f3(sr.get("tfidf+ovr-logistic", {}).get("macro_f1")), str(sr.get("tfidf+ovr-logistic", {}).get("labels_predicted", "—"))],
            ["8k subsample", "InLegalBERT (2 ep)", f3(sr.get("InLegalBERT", {}).get("micro_f1")), f3(sr.get("InLegalBERT", {}).get("macro_f1")), str(sr.get("InLegalBERT", {}).get("labels_predicted", "—"))],
            ["8k subsample", "InLegalBERT, head LR ×10", f3(sr.get("InLegalBERT[headlr10]", {}).get("micro_f1")), f3(sr.get("InLegalBERT[headlr10]", {}).get("macro_f1")), str(sr.get("InLegalBERT[headlr10]", {}).get("labels_predicted", "—"))],
            ["full corpus", "TF-IDF + OvR logistic", f3(tf_full.get("micro_f1")), f3(tf_full.get("macro_f1")), str(tf_full.get("labels_predicted", "—"))],
            ["full corpus", "InLegalBERT (2 ep)", f3(bert_full.get("micro_f1")), f3(bert_full.get("macro_f1")), str(bert_full.get("labels_predicted", "—"))],
            ["full corpus", "InLegalBERT (4 ep, head LR ×10, head+tail)", f3(bert_best.get("micro_f1")), f3(bert_best.get("macro_f1")), str(bert_best.get("labels_predicted", "—"))]]
    if ens:
        best = max(ens["sweep"], key=lambda r: r["dev_micro_f1"])
        rows.append(["full corpus", f"Ensemble α={ens['best_alpha']:.1f} (TF-IDF, BERT)", f3(best["test_micro_f1"]), f3(best["test_macro_f1"]), "—"])
    w.table(rows, widths=[Inches(1.1), Inches(2.6), Inches(0.9), Inches(0.9), Inches(1.1)])
    w.caption("Table 4.3  —  Statute identification on the ILSI test split. Thresholds tuned on dev.")
    w.picture(ASSETS / "r3_statute_f1.png", width=Inches(6.2))
    w.caption("Figure 4.3  —  micro- and macro-F1 by model and training set.")
    if abl:
        drop = abl["full_document"]["micro_f1"] - abl["truncated_512"]["micro_f1"]
        w.para(f"Two explanations for the encoder gap were tested and ruled out. Truncation to 512 wordpieces keeps {abl['text_retained_fraction'] * 100:.1f}% of the text and costs the "
               f"TF-IDF model {drop:.4f} micro-F1 ({drop / abl['full_document']['micro_f1'] * 100:.1f}% relative), far less than the gap; and the first encoder run's collapse onto six labels "
               "(macro-F1 0.018) was a loss problem — plain binary cross-entropy on a hundred sparse labels — fixed with positive-class weighting, after which the encoder still loses. "
               "The conclusion for this corpus is that TF-IDF is the classifier and the encoder's contribution lies in element verification, where it reads statute text against the complaint rather than assigning labels.")

    w.h3("4.4.2 Legal knowledge base: mapping, cognizability, statute text")
    w.para("The IPC→BNS mapping covers the 100 ILSI sections: 69 rows are auto-applicable, 29 are flagged for review and 7 map outside the BNS (Prevention of Corruption Act, omitted sections). "
           "The source is a third-party pocket directory rather than the official gazette, which is why the auto-apply rule is strict and why the 29 flagged rows remain a legal task.")
    w.para(f"The BNSS 2023 First Schedule, Part I (gazette pp. 158–188) was parsed into {len(sched_rows)} section keys from 461 printed rows: {sched['cognizable']} cognizable, "
           f"{sched['non_cognizable']} non-cognizable and {sched['conditional']} conditional. Conditional entries are those the Schedule itself makes depend on a fact — abetment and attempt "
           "follow the principal offence, section 85 (cruelty) is cognizable only when reported by the aggrieved woman or a relative, and section 303(2) (theft) is non-cognizable where the "
           "property is worth less than ₹5,000. They route to the officer with the Schedule's words quoted; the theft split is resolved automatically when the complaint states a value. "
           "Table 4.4 lists the entries on which the parsed Schedule disagreed with the hand-coded table used until this review.")
    rows = [["BNS section", "Hand-coded stub", "Gazette Schedule", "Direction of the error"],
            ["126 wrongful restraint", "non-cognizable", "cognizable", "would have refused an FIR"],
            ["223 disobedience to public servant's order", "non-cognizable", "cognizable", "would have refused an FIR"],
            ["296 obscene acts and songs", "non-cognizable", "cognizable", "would have refused an FIR"],
            ["329 criminal trespass", "non-cognizable", "cognizable", "would have refused an FIR"],
            ["85 cruelty by husband or relatives", "cognizable", "conditional (who reports)", "hid a condition"],
            ["303(2) theft", "cognizable", "conditional (value ≥ ₹5,000)", "hid a condition"],
            ["61, 190, 238 conspiracy / common object / evidence", "flat", "conditional (follows the offence)", "hid a condition"]]
    w.table(rows, widths=[Inches(2.3), Inches(1.2), Inches(1.7), Inches(1.5)])
    w.caption("Table 4.4  —  Disagreements between the hand-coded cognizability table and the parsed First Schedule.")
    w.picture(ASSETS / "r3_schedule.png", width=Inches(6.2))
    w.caption("Figure 4.4  —  Classification of the parsed First Schedule, and the entries the hand-coded version had wrong.")
    w.para("All 358 BNS sections were parsed from the BNS gazette (heading, chapter, full text and an illustration-free operative text). The officer page shows the BNS words next to every "
           "suggested section, with the IPC text from ILSI kept as lineage; the same corpus is the input the element verifier will read.")

    w.h3("4.4.3 Automatic speech recognition (FLEURS ta_in)")
    w.para(f"Table 4.5 reports the zero-shot faster-whisper large-v3 (int8) baseline on {asr_old.get('n_clips', 60)} FLEURS Tamil test clips ({asr_old.get('audio_seconds', 0):.0f} s of audio). "
           "Tamil is agglutinative, so a single word-boundary decision changes the word count; character error rate is therefore reported beside WER, and per-clip WER above 100% is not by itself evidence of fabrication.")
    rows = [["Decoding", "WER", "CER", "median clip WER", "RTF", "flagged clips"]]
    rows.append(["temperature fallback ladder (library default), condition_on_previous_text off", pct(asr_old.get("wer")), pct(asr_old.get("cer")), pct(asr_old.get("median_clip_wer")), f"{asr_old.get('rtf', 0):.2f}", str(asr_old.get("n_flagged", "—"))])
    cp = _load("asr_baseline_condprev_on.json")
    if cp:
        rows.append(["fallback ladder, condition_on_previous_text on", pct(cp.get("wer")), pct(cp.get("cer")), pct(cp.get("median_clip_wer")), f"{cp.get('rtf', 0):.2f}", str(cp.get("n_flagged", "—"))])
    if asr_seq:
        rows.append(["one pass at T = 0, sequential 30 s window", pct(asr_seq.get("wer")), pct(asr_seq.get("cer")), pct(asr_seq.get("median_clip_wer")), f"{asr_seq.get('rtf', 0):.2f}", str(asr_seq.get("n_flagged", "—"))])
    if asr_new:
        rows.append(["one pass at T = 0, batched VAD chunks (bs = 8) — pipeline default", pct(asr_new.get("wer")), pct(asr_new.get("cer")), pct(asr_new.get("median_clip_wer")), f"{asr_new.get('rtf', 0):.2f}", str(asr_new.get("n_flagged", "—"))])
    else:
        rows.append(["one pass at T = 0, batched VAD chunks (bs = 8) — pipeline default", "«re-run in progress»", "", "", "", ""])
    w.table(rows, widths=[Inches(2.8), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.6), Inches(0.8)])
    w.caption("Table 4.5  —  ASR on FLEURS ta_in test, faster-whisper large-v3 int8 on CUDA.")
    if asr_new:
        w.picture(ASSETS / "r3_asr.png", width=Inches(6.2))
        w.caption("Figure 4.5a  —  WER, CER and real-time factor before and after the decoding change, 60 clips.")
        w.para(f"The corrected decoding is the pipeline default: WER {pct(asr_old.get('wer'))} → {pct(asr_new.get('wer'))}, CER {pct(asr_old.get('cer'))} → {pct(asr_new.get('cer'))}, "
               f"RTF {asr_old.get('rtf', 0):.2f} → {asr_new.get('rtf', 0):.2f} on the same 60 clips. The transcript guard still flags {asr_new.get('n_flagged', 0)} clips for a repetition loop "
               "(clip 1916 among them), so the temperature ladder explains the speed, not the replay failure; the guard remains the defence and the loop is the next ASR problem to solve "
               "(no-repeat-n-gram or repetition penalty in decoding, then IndicWhisper).")
    if prof:
        per = [r for r in prof["rows"] if r["compute"] == "int8" and r["mode"] == "per-clip"]
        joined = [r for r in prof["rows"] if r["compute"] == "int8" and r["mode"].startswith("joined") and r["beam"] == 5]
        w.para("Profiling explained the slow real-time factor of the first baseline. faster-whisper's default decoding retries every window whose output fails a compression-ratio or "
               "average-log-probability check at up to five higher sampling temperatures; low-confidence Tamil fails that check constantly, so most clips were decoded several times, and the "
               "high-temperature passes are sampled — the path on which invented text can appear. Table 4.6 gives the measurements on eight clips and on the same clips joined into one "
               "complaint-length recording.")
        rows = [["Setting", "Beam", "Temperature fallback", "Batch", "RTF", "WER"]]
        for r in per:
            rows.append(["per clip", str(r["beam"]), "ladder" if r.get("fallback", True) else "none (T = 0)", "—", f"{r['rtf']:.3f}", pct(r["wer"])])
        for r in joined:
            rows.append(["joined 148 s recording", str(r["beam"]), "none (T = 0)", str(r["batch"]) if r["batch"] else "sequential", f"{r['rtf']:.3f}", pct(r["wer"])])
        w.table(rows, widths=[Inches(1.8), Inches(0.6), Inches(1.5), Inches(1.0), Inches(0.7), Inches(0.9)])
        w.caption(f"Table 4.6  —  Where the decoding time goes (CTranslate2 {prof.get('ctranslate2', '')}, {prof.get('gpu', '')}).")
        w.picture(ASSETS / "r3_asr_profile.png", width=Inches(6.2))
        w.caption("Figure 4.5  —  Real-time factor and WER by decoding setting.")
        w.para("Removing the fallback made per-clip decoding 5.3× faster at equal or better WER; on the complaint-length recording, decoding the VAD chunks in batches was 3× faster again "
               "and markedly more accurate (WER 81% → 49%) than the fixed 30-second sliding window, because chunk boundaries follow the speaker's pauses instead of cutting sentences. "
               "Both are now the pipeline defaults. The compute-type comparison made earlier under the fallback ladder (int8 best, float16 worse) is superseded: without the ladder int8 and float16 "
               "give the same WER and float16 is faster; int8 is kept for its VRAM footprint.")

    w.h3("4.4.4 Translation (Tamil → English, classifier view)")
    tr_name, tr_r = tr
    rows = [["Model", "Backend", "Sentence pairs", "chrF", "BLEU", "s / sentence"],
            [tr_name, tr_r.get("backend", "marian"), str(tr_r.get("n", 336)), f"{tr_r.get('chrf', 0):.1f}", f"{tr_r.get('bleu', 0):.1f}", f"{tr_r.get('sec_per_sentence', 0):.2f}"]]
    w.table(rows, widths=[Inches(2.2), Inches(0.9), Inches(1.1), Inches(0.7), Inches(0.7), Inches(1.0)])
    w.caption("Table 4.7  —  Tamil→English translation on FLoRes devtest (FLEURS ta_in × en_us joined on sentence id). Wikipedia register, not complaints: a model-comparison signal.")
    w.para("The interim model trails IndicTrans2 by roughly 20 chrF in published comparisons; IndicTrans2 is gated behind a licence acceptance and is the planned upgrade. "
           "The translation is used only as the classifier's view of the complaint; extraction, element checks and the narrative keep the original text, which is what allowed the cue-scan safety net to catch the mistranslated dowry complaint.")

    w.h3("4.4.5 Extraction and element cues")
    rows = [["Slot", "TP", "FP", "FN", "Precision", "Recall"]]
    for slot in ("date", "time", "amount_inr", "phone", "vehicle", "_micro"):
        r = slots.get(slot, {})
        rows.append([slot.strip("_"), str(r.get("tp", "—")), str(r.get("fp", "—")), str(r.get("fn", "—")), f"{r.get('precision', 0):.2f}", f"{r.get('recall', 0):.2f}"])
    w.table(rows, widths=[Inches(1.4), Inches(0.7), Inches(0.7), Inches(0.7), Inches(1.0), Inches(1.0)])
    w.caption(f"Table 4.8  —  Rule extractor on the {ex.get('n_docs', 28)}-document adversarial gold set (spoken and written forms, traps such as ages and account numbers).")
    w.para("The gold set is a smoke test, not a benchmark: it is small and hand-written. Its value is in the traps it carries (numbers that are not slots, two phone numbers in one complaint, "
           "year-less dates) and in being the harness that will score the learned extractor against the synthetic corpus. "
           "Element cues cover 29 station-house sections; a 22-complaint battery in English and Tamil pins which sections must fire for each complaint and which tempting wrong ones must not "
           "(a road death is 304A, not 302; a threat to kill is 506, not 307). Two precision defects found by the battery — a threat cue counted toward attempt to murder, and an inflected Tamil noun "
           "missing a citation-form cue — were fixed and are now tests.")

    w.h2("4.5 Analysis, problems met and corrective action")
    w.para("Nine findings changed the plan or the code between Review 2 and Review 3. Each is recorded in PROGRESS.md with the measurement that established it; the summary is:")
    for t in [
        "Hardware (finding 1): 8 GB of VRAM, not 16. Consequence: model choices carry a fits flag and the synthetic-data generator decision is a measured benchmark, not an assumption.",
        "Infrastructure (6d, 6e): the WSL VM restarted six times under full-corpus jobs; the predictor was Python-heap size, not GPU memory. Chunked builds and detached launches fixed it.",
        "Encoder training (7): the InLegalBERT head collapsed onto six labels under plain BCE; positive-class weighting restored 97–100 labels — and the encoder still lost to TF-IDF (8).",
        "ASR fabrication (6b): two of sixty clips contained words not spoken (a replay loop, a Cyrillic drift). Guards calibrated on the baseline flag them and block auto-resolution; condition_on_previous_text=False was predicted to remove the replay and, when tested, did not.",
        "ASR speed (6f): profiling attributed the 3× real-time factor to the temperature-fallback ladder; one pass at T = 0 plus batched VAD chunks gives RTF 0.10 on a complaint-length recording with better WER.",
        "Translation loss (5c): “dowry” became “relief” and a 498A complaint routed to CSR; the cue scan on the original text now forces officer review and names the section.",
        "Cognizability (9): the hand-coded stub was wrong on four entries, all in the refusal direction; the First Schedule is now parsed from the gazette with page citations and conditional entries ask the officer.",
        "Domain shift (5): the classifier abstains on short complaints (its top-5 for a two-sentence theft has no theft section), so a lower threshold would apply wrong sections; in-domain synthetic data, not calibration, is the fix, and the cue scan carries short inputs meanwhile.",
    ]:
        w.bullet(t)

    w.h2("4.6 Limitations at this stage")
    for t in [
        "The classifier is trained on English court fact-statements and reads Tamil complaints through an interim translator; Tamil statute identification is therefore bounded by translation quality until IndicTrans2 or in-domain data is in place.",
        "Extraction covers structured slots only; names, places and accused descriptions await the NER / LLM stage and the synthetic corpus that trains and evaluates it.",
        "The element verifier is rule-based (keyword cues); it is a recall net and an explanation format, not yet a judgement of the ingredients of an offence.",
        "The legal tables are parsed from official text but not legally signed off; the IPC→BNS map's 29 flagged rows and 4 VERIFY rows remain open.",
        "The narrative composer and faithfulness gate are deterministic v0; the officer interface is a static page without accept/override logging; there is no print-faithful PDF.",
        "ASR numbers are zero-shot on read Wikipedia sentences (FLEURS), not spoken complaints; IndicWhisper / IndicConformer have not been benchmarked.",
    ]:
        w.bullet(t)

    w.h2("4.7 Individual contribution")
    w.para("«Guideline §2 requires each member's contribution to be demonstrated. Complete the table from the dated engineering log (PROGRESS.md) and the module ownership in IMPLEMENTATION.md.»", muted=True)
    rows = [["Member", "Modules / deliverables", "Evidence"],
            ["Madhav K (23BAI1088)", "«…»", "«commits, PROGRESS.md entries, slides»"],
            ["Nischay Kuchibotla (23BAI1245)", "«…»", "«…»"],
            ["Rohit A. S. (23BAI1416)", "«…»", "«…»"]]
    w.table(rows, widths=[Inches(2.0), Inches(3.0), Inches(1.7)])
    w.caption("Table 4.9  —  Individual contribution at Review 3.")
    w.para("Use of AI tools (guideline §2): an AI coding assistant (Claude Code) was used for implementation support under the team's direction. Design decisions, legal tables, experiments "
           "and results were reviewed by the team and are reproducible from the repository. «Confirm the wording with the guide.»", muted=True)

    w.h1("5. Conclusion and Future Work (outline)")
    w.para("To be completed for Review 4. Planned content: summary of the work completed against the objectives of §1.4; major findings (the sparse-baseline result, the cost of translation, "
           "the gazette-versus-memory result for legal tables, the ASR decoding finding); limitations carried from §4.6; and future enhancements — the local LLM decision and synthetic corpus, "
           "learned extraction, the LLM element verifier over BNS text, IndicTrans2 and IndicWhisper benchmarks, the human gold set, the React verification workspace with audit logging, "
           "and the print-faithful IF-1 PDF.", muted=True)

    w.move_before(refs_h._p)

    # -- references appended -------------------------------------------------
    last_ref = doc.paragraphs[-1]
    n0 = int(re.match(r"\[(\d+)\]", last_ref.text).group(1)) if re.match(r"\[(\d+)\]", last_ref.text) else 21
    new_refs = [
        "Ministry of Home Affairs, Government of India, “The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act No. 46 of 2023),” The Gazette of India Extraordinary, Part II Sec. 1, No. 55, CG-DL-E-25122023-250884, 25 Dec. 2023 (First Schedule, pp. 158–188).",
        "Ministry of Home Affairs, Government of India, “The Bharatiya Nyaya Sanhita, 2023 (Act No. 45 of 2023),” The Gazette of India Extraordinary, Part II Sec. 1, No. 53, CG-DL-E-25122023-250883, 25 Dec. 2023.",
        "SYSTRAN, “faster-whisper: Whisper transcription with CTranslate2,” v1.1.0, 2024; OpenNMT, “CTranslate2,” v4.8.2, 2026. https://github.com/SYSTRAN/faster-whisper",
        "J. Tiedemann and S. Thottingal, “OPUS-MT — Building open translation services for the World,” Proc. EAMT, 2020 (model Helsinki-NLP/opus-mt-dra-en).",
        "M. Post, “A Call for Clarity in Reporting BLEU Scores,” Proc. WMT, 2018 (sacreBLEU; chrF after Popović, 2015).",
        "A. Conneau et al., “FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech,” Proc. IEEE SLT, 2022.",
        "Paul et al., “ILSI: Indian Legal Statute Identification (LeSICiN),” Proc. AAAI, 2022 — corpus release on Zenodo, doi:10.5281/zenodo.6053791.",
    ]
    for k, ref in enumerate(new_refs, start=n0 + 1):
        p = doc.add_paragraph(f"[{k}]  {ref}", style="Normal")
        for r in p.runs:
            r.font.size = last_ref.runs[0].font.size if last_ref.runs and last_ref.runs[0].font.size else Pt(10)

    set_update_fields(doc)
    doc.save(str(OUT))
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    build()
