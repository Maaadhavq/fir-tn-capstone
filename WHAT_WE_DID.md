# What we did — in plain words

> For Madhav, before Review 3 (16 Sep 2026). Everything here is true of the repository as it stands.
> Exact numbers and file paths are in `PROJECT_CONTEXT.md`; this file is the plain-language version.

---

## 1. The problem we chose

When someone reports a crime at a Tamil Nadu police station, an officer has to listen to them (usually in
Tamil, often mixed with English), write it down, decide which sections of the law apply, decide whether it
is serious enough for an FIR at all, and fill a fixed government form called the IF-1 (15 numbered items).
It is slow, and the section-picking depends a lot on which officer you get.

## 2. What we built

A local program that listens to a spoken complaint and produces a **draft** FIR for the officer to check.

Three rules we never break, and you should say them out loud in every review:

1. **It drafts, it never files.** The officer is the author of record. Nothing in the code can mark a record
   as "verified" — there is even a test that fails if someone adds a way to do it.
2. **Everything runs on the laptop.** No cloud, no internet at run time. Complaints are personal data.
3. **The form is fixed.** We do not design a document. We fill the standard IF-1. The only free text is the
   short "what happened" paragraph, and every sentence in it must trace to a fact we extracted.

## 3. How it works — the five steps

Think of a conveyor belt. A complaint enters as sound and leaves as a filled form.

**Step A — Hear it (speech recognition).** We use Whisper, a speech model, running on the laptop GPU. Two
things we added on top: a *guard* that notices when Whisper starts making text up (it repeats itself or
switches to a foreign alphabet — this happened on 2 of 60 test clips), and a *number converter* that turns
spoken Tamil numbers into digits ("ஐம்பதாயிரம் ரூபாய்" becomes ₹50,000, "எட்டாம் தேதி" becomes the 8th).

**Step B — Pull out the facts.** Rules find dates, times, rupee amounts, phone numbers and vehicle plates in
the transcript, and remember exactly where each one came from (so the officer can click a value and see the
words behind it). If a value is ambiguous — two phone numbers, two dates — it is left blank rather than
guessed. This is the "floor" version; a learned version comes in semester 2.

**Step C — Find the law.** This is the research part.
- A classifier trained on 66,000 Indian court documents suggests which IPC sections fit.
- A conversion table turns old IPC numbers into new BNS 2023 numbers. It is applied automatically only when
  the mapping is certain; uncertain rows are shown to the officer instead.
- For each suggested section, the system checks the legal "ingredients" (for theft: took something, it was
  movable property, from someone else, without consent) and shows which ones the complaint actually
  evidences, quoting the words. It never says "no" — a missing ingredient is a question for the officer.
- The same ingredient check runs on the *original Tamil* as a safety net, because the classifier only reads
  English and translation can lose words.

**Step D — FIR or not?** Indian law splits offences into cognizable (police can register an FIR and
investigate) and non-cognizable (goes to a different register). The official list is the BNSS First
Schedule. We parsed it from the government gazette PDF. Some entries are conditional — theft under ₹5,000
is non-cognizable, cruelty is cognizable only if the victim or a relative reports it — and the system routes
those to the officer with the Schedule's own words instead of guessing.

**Step E — Fill the form and show it.** The 15 items are filled deterministically; a short narrative is
written from the facts; the whole thing appears on a local web page where the officer can type, upload a
recording, or speak into the microphone and get a draft in a few seconds. A "Mark verified" button exists
and is deliberately greyed out. Every draft is logged (ID, route, sections — never the text).

## 4. How we know it works — the numbers and where they come from

We tested each step separately on public data. Each test has a dataset, a scoring rule and a purpose.

| Step | Dataset | Score | What it means |
|---|---|---|---|
| Law sections | 13,039 unseen court documents (ILSI) | F1 **0.70** (simple TF-IDF model) vs **0.53** (InLegalBERT, the fancy legal model we planned) | The simple model won by a wide margin; combining both added nothing. We use the simple one. |
| Speech | 60 public Tamil clips (FLEURS) | Word error **51%**, character error **15%**, speed 0.46× real time | Last week it was 57% / 22% and slower than real time. We found the library was re-transcribing uncertain audio up to six times; switching that off made it 4× faster and more accurate. |
| Fabrication guard | same 60 clips | **2 of 60** clips caught | We tried to stop the repetition inside the decoder; it fixed those two clips but made every other clip worse, so we kept the guard instead. |
| Translation (Tamil → English, for the classifier only) | 336 sentence pairs | chrF **35** | An interim model; the best Indian model needs a gated download and would roughly double this. |
| Fact extraction | 28 complaints we wrote, 63 values | **63/63, zero wrong values** | Our own test set, not a benchmark. Zero wrong values is the property a form needs. |
| Legal tables | the government gazette | **438** cognizability entries; **358** BNS section texts; IPC→BNS map 100 rows (69 automatic, 29 held) | Parsing the gazette found **4 errors** in the hand-written table we had used before — all of the "would have wrongly refused an FIR" kind. |
| Code health | — | **376 automated tests**, run in ~25 s | Legal rules are written as tests. |

**How to read WER 51%:** high, but honest. It is measured on people reading Wikipedia sentences, not on
police speech, with no Tamil-specific training. Tamil glues words together, so word error exaggerates —
character error (15%) is the fairer number. The fix is a Tamil-trained speech model, which is the first
thing on the semester-2 list.

## 5. What we learned — the findings (this is what panels like)

Fourteen things went wrong or surprised us; each is written up with the measurement and the fix.

1. **The laptop GPU is 8 GB, not the 16 GB our plan assumed.** Everything we built fits; the large
   language model we planned for generating training data does not. Recorded, not hidden.
2. **The simple model beat the fancy one.** InLegalBERT first collapsed to predicting only 6 of 100
   sections (a loss-function problem, fixed), then still lost to TF-IDF after every improvement we tried.
3. **Whisper invents text sometimes.** We built a guard, calibrated it on real clips, and tested that
   blocking the decoder's repetition instead would hurt more than it helps.
4. **Translation lost the word "dowry"** — it came out as "relief" — and a dowry-cruelty complaint was
   routed to the non-serious path. That is why the ingredient check now runs on the original Tamil.
5. **Our hand-written cognizability table was wrong in 4 places.** Legal tables must come from the
   official text with page citations, not from memory. Now they do.
6. **The speech decoder was 3–4× slower than it should be** because of a library default. One measured
   change fixed speed and accuracy together.
7. **The development VM crashed under big jobs** — we learned to chunk every large computation and run
   long jobs detached. (Separately, the laptop hard-reset twice on 15 Sep under sustained GPU load — keep
   review-day GPU use to short demo bursts, on mains power.)

The one-line story: *a cheap baseline beat the domain model, the law had to come from the gazette, and
every safety guard exists because we reproduced a real failure first.*

## 6. What is NOT done (say this before they ask)

- The learned extraction (NER / LLM), the LLM that writes the narrative and verifies legal ingredients,
  synthetic training data, the React officer interface, the printable PDF form, fine-tuning — semester 2.
- The classifier struggles on very short complaints (it learned from long court text). The ingredient
  safety net covers this; in-domain data is the real fix.
- The legal tables have not been checked by a lawyer.
- The translator is a stop-gap model.

## 7. Things you are supposed to know

**Team and course.** Madhav K (23BAI1088), Nischay Kuchibotla (23BAI1245), Rohit A. S. (23BAI1416); guide
Dr. Shivaranjani; BCSE497J Project-I, VIT Chennai, SCOPE. Review 3 (panel) 16 Sep; Review 4 (guide)
12–16 Oct; final panel 21 Oct.

**How the work actually happened.** The implementation from 20 Aug to 15 Sep was done by you with an AI
coding assistant (Claude Code). The course guidelines say significant AI use must be acknowledged. Decide
with your guide how to word it; a safe line is on slide 15. The panel also grades each member's
*demonstrated* contribution — agree the split honestly with Nischay and Rohit and make sure each can explain
their part. Do not have a slide claim something the commit log cannot back up.

**Two blanks only you can fill.** The Review 2 panel comments and what you did about each (slide 3 — one
mark is only for this), and the contribution table (slide 16).

**Where things live.**
- GitHub: https://github.com/Maaadhavq/fir-tn-capstone (public; `gh repo edit --visibility private` if you
  prefer to add collaborators instead).
- `PROJECT_CONTEXT.md` — the long, exact version of this file, for teammates and LLMs making slides.
- `PROGRESS.md` — the dated engineering log with every finding.
- `docs/REVIEW3_DEMO.md` — pre-flight checklist, six demo steps, fallbacks, likely questions.
- `docs/REVIEW3_SCRIPT.md` — what to say, slide by slide.
- `Review3_Presentation.pptx/.pdf` — the deck; `Review3_Report.docx` — Chapters 1–4 if asked.
- `artifacts/reports/RESULTS.md` — every number.

**How to run it.** Everything runs in WSL (Ubuntu). Start the server:
`wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && FIR_PREWARM=1 bash scripts/run.sh serve"`
then open http://127.0.0.1:8000/ — the page has sample buttons and a microphone button. Tests:
`bash scripts/run.sh test -q`. If the GPU is busy or the server fails, the CLI gives the same answer:
`python -m fir draft "…"` from `src/`.

**Demo-day cautions.** Laptop on mains power. Allow the microphone in Edge once before the panel. Keep
FLEURS clip `12583250098003224463.wav` on the desktop — uploading it shows the fabrication guard firing
live. If nothing runs, the PDF has the same screens.

**Review 4 needs two things now.** A publication venue agreed with your guide (Review 4 gives 2 marks only
for *proof of submission*), and a near-final report — the Chapter 4 draft already exists to build on.

## 8. Vocabulary

FIR — First Information Report · IF-1 — the fixed TN FIR form · BNS / BNSS — the 2023 laws that replaced
IPC / CrPC · cognizable — police can register an FIR without a court order · CSR — register for
non-cognizable complaints · ILSI — the court-document dataset · WER / CER — word / character error rate ·
RTF — processing time ÷ audio length (below 1 = faster than real time) · F1 — accuracy measure balancing
precision and recall · TF-IDF — a simple word-counting model · InLegalBERT — a large model pre-trained on
Indian legal text · LangGraph — the framework that chains the five steps.
