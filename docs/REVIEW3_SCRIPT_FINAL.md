# Speaking script for `Review3_Presentation_Final.pptx` (15 slides, ≈8 min + demo)

Plain spoken text. [Brackets] are notes to you, not to be read. ▶ marks a good hand-over point.

---

**Slide 1 — Title**
Good morning. We are presenting Review 3 of *Speech-Driven FIR Drafting for Tamil Nadu Police* — Madhav,
Nischay and Rohit, guided by Dr. Sivaranjani. Today we show implementation progress and intermediate
results; the system now runs end to end.

**Slide 2 — Project scope**
The input is a spoken complaint in Tamil or Tamil-English. The system transcribes it, extracts the facts,
suggests the applicable BNS sections, and helps decide whether the matter is cognizable. The output is a
*draft* IF-1 — the standard Tamil Nadu FIR form — filled only from facts we can support. The control point
is the officer: every suggestion is reviewed by them, and they remain the author of record. Nothing is
filed automatically.

**Slide 3 — Progress since Review 2**
Five things work end to end: the complaint pipeline, Tamil speech recognition, structured fact extraction,
statute identification, and IF-1 draft generation. Four are working in a first version and being
improved: Tamil-to-English translation, element-level legal checks, the officer review screen, and audit
and provenance display. Remaining for semester two: extraction of names and places, a stronger translation
model, a legal review of our mappings, a gold set of real complaints, and the final user evaluation.
Against our plan of 27 milestones, 10 are working, 11 partial, 6 not started — roughly the fifty percent
Review 3 expects, with the whole path running rather than half the stages finished.

**Slide 4 — Current pipeline**
Left to right: complaint audio or text, speech recognition, fact extraction, statute and legal checks,
IF-1 draft, officer review. Today the input can be typed text or a recording made in the browser; the
output is a draft IF-1 with suggested sections, the evidence for each, and warnings. The safety boundary
is fixed: there is no automatic verification and no filing — not even a code path for it.

▶ **Slide 5 — Speech recognition** [Rohit]
We run Whisper large-v3 locally on the laptop GPU. We added quality checks that flag transcripts that
repeat themselves or drift into another script — the model invented text on 2 of 60 test clips — and a
Tamil normaliser that turns spoken numbers, dates, times and amounts into digits.
The table compares last week's setting with the current one on 60 public Tamil clips: word error rate
57.2 to 51.4 percent, character error rate 22.4 to 15.0, and processing time from 1.8 times real time to
0.46 — faster than real time. One measured change did this: the library was re-transcribing uncertain
audio up to six times at increasing randomness; we turned that off and decode speech chunks in batches.
Character error rate is the more informative number for Tamil, because Tamil joins words together and one
boundary decision changes the word count. The two flagged clips remain flagged — we tested a decoder-level
fix, it made every other clip worse, so we kept the guard.

▶ **Slide 6 — Fact extraction and translation** [Nischay]
Extraction pulls dates, times, rupee amounts, phone numbers and vehicle numbers from the transcript. It
fills a field only when the value is unambiguous, and records the source text for every value, so the
officer can trace it. On our smoke test of 28 complaints it found 63 of 63 fields with no false
positives — and zero false positives is the property a form needs: a blank prompts a question, a wrong
value gets trusted.
Translation exists for one reason: the statute classifier reads English. The current model scores chrF
35.2 on 336 sentence pairs — an interim component. The original Tamil is kept for extraction and legal
evidence, and a bilingual cue check on the original text sends uncertain legal cases to the officer, so a
translation error cannot silently downgrade a serious complaint.

▶ **Slide 7 — Statute identification results** [Madhav]
This is the research core. Training and testing use ILSI — Indian court documents labelled with IPC
sections — with 13,039 test documents and 100 section labels. The TF-IDF classifier reaches micro-F1
0.700 and macro-F1 0.581. The neural legal model, InLegalBERT, which we proposed at Review 2, reached
0.526 after every improvement we could apply; combining the two improved micro-F1 by only 0.0006. So the
decision is to use the TF-IDF classifier and give the neural model a different job later — verifying
legal ingredients rather than picking sections. We report this as measured, not as we planned it.

**Slide 8 — Legal knowledge and routing**
Three tables carry the law. The text of all 358 BNS sections, parsed from the official notification, is
shown with every suggestion. The BNSS First Schedule — which decides cognizable versus non-cognizable —
was parsed from the gazette into 438 section keys: 277 cognizable, 135 non-cognizable, and 26 conditional,
such as theft under five thousand rupees or cruelty depending on who reports; conditional cases stay with
the officer with the Schedule's own words. The IPC-to-BNS mapping covers our 100 labels: 69 apply
directly, 29 require review, 7 fall outside the BNS. Only high-confidence mappings are ever applied
automatically. Parsing the gazette also caught four errors in the hand-written table we used earlier —
which is why we no longer type legal tables from memory.

**Slide 9 — Legal justification and safeguards**
For each candidate section the officer sees the official text, the required ingredients of the offence,
the complaint words that support each ingredient, and the ingredients still unresolved, marked as
unclear. Three safety behaviours: a missing word is never treated as proof that an element is absent;
cues are checked in both Tamil and English against the original complaint; and a cognizable or
conditional section the classifier missed sends the case to officer review. Element checks currently
cover 29 commonly encountered sections. Every suggestion is advisory and editable.

**Slide 10 — FIR draft and officer review**
The officer's workflow has four steps: review the complaint evidence — transcript and extracted facts
stay visible; review the legal suggestions with their evidence and open questions; review the IF-1
fields — only supported values enter the fixed 15-item form; and edit or reject. No component can mark a
draft as verified or submit it to the police system.
[Run the demo here — six steps in docs/REVIEW3_DEMO.md: typed theft, remove the value, dowry case,
spoken Tamil sample, live microphone, upload clip 1916 to show the guard.]

**Slide 11 — Intermediate results**
One table, every row with its dataset: speech 51.4 percent WER and 15.0 CER on 60 clips; translation
chrF 35.2 on 336 pairs; extraction 63 of 63 with zero false positives on 28 complaints; statute
identification micro-F1 0.700 and macro-F1 0.581 on 13,039 documents; the mapping 69 direct, 29 review,
7 outside; and 376 automated tests passing across 18 files. All numbers are generated from result files
in the repository, none are typed in.

**Slide 12 — Testing and current limitations**
Testing: end-to-end fixtures for typed and audio complaints, schema consistency and fixed-form output,
Tamil number normalisation and transcript warnings, and the legal routing rules for unknown and
conditional cases — the safety rules are tests, so breaking one fails the build.
Limitations, stated plainly: the speech error rate is still high on unseen Tamil audio; translation is
interim; extraction does not yet cover names and locations; the legal tables need expert review; and
evaluation on real complaint recordings has not started.

**Slide 13 — Report progress and team responsibilities**
The introduction, literature review and methodology chapters are updated; the implementation and results
chapter has an initial draft; conclusion and final evaluation remain. Responsibilities: Madhav — statute
identification, IPC-to-BNS mapping, legal routing and integration; Rohit — speech recognition,
transcript quality and Tamil normalisation; Nischay — fact extraction, data preparation and named-entity
extraction. [Add one sentence acknowledging AI-assisted coding if your guide agrees: "Implementation was
carried out with the assistance of an AI coding assistant; all design decisions, experiments and their
interpretation were reviewed and are owned by the team."]

**Slide 14 — Remaining work and timeline**
Sixteenth to thirtieth September: improve translation, compare Tamil speech baselines, add name and
location extraction. First to eighth October: complete element verification, review the legal mappings,
prepare a small Tamil complaint gold set. Ninth October to Review 4: end-to-end evaluation, complete the
report, rehearse the demonstration, and prepare publication evidence. Review 4 is 12 to 16 October.

**Slide 15 — Summary**
A working system: audio or text now produces a draft IF-1 through one integrated pipeline. Measured
progress: statute identification at 0.700 micro-F1; extraction found all 63 fields with no false
positives; speech four times faster and more accurate than last week. A known limitation: Tamil speech
and translation quality still constrain the full pipeline. The safety position: the officer reviews the
evidence, the suggestions and the form; the system cannot file or verify an FIR. Thank you — questions,
and the live demo.

---

## Likely questions — one breath each

- *Why does the classifier show nothing on a short complaint?* It learned from long court text; on a
  two-line complaint its top five don't include theft, so a lower threshold would apply wrong sections.
  The cue check carries short inputs as a question; in-domain data is the fix.
- *Is 51 percent WER usable?* As decision support with provenance and guards, yes; as a transcript to
  file, no. It's zero-shot on read Wikipedia Tamil; a Tamil-fine-tuned model is next.
- *Why not the BERT model you proposed?* Measured: 0.526 versus 0.700; the ensemble adds nothing.
- *Who verified the legal tables?* Nobody yet — parsed from the gazette with page citations and
  spot-checked in tests; a legal read-through is on the plan.
- *Can it file an FIR?* No path exists; a test fails if one appears.
- *What was AI's role?* Coding assistance, acknowledged; decisions and interpretation are ours.
