# Review 3 — speaking script (≈9 minutes + 6-minute demo)

Slide numbers match `Review3_Presentation.pptx`. Speak the plain text; the bracketed notes are for you.
Hand-over points between members are marked ▶ — decide tonight who takes which block.

---

**Slide 1 — Title**
Good morning. We are presenting Review 3 of *Speech-Driven Drafting of First Information Reports for the
Tamil Nadu Police*. Our guide is Dr. Shivaranjani. In one line: a citizen speaks a complaint in Tamil,
and the system produces a *draft* FIR in the standard IF-1 form for a police officer to verify. It
drafts; it never files.

**Slide 2 — Rubric coverage**
This slide maps the seven Review 3 parameters to the slides that answer them, so you can check us as we
go. Everything with a number today comes from a results file in the repository; nothing is estimated.

**Slide 3 — Follow-up on Review 2 feedback**
[Fill before the review. Read each panel comment, then one line of action.]
At Review 2 the panel asked us … — we did … .
Two changes we made ourselves are also here: the development GPU turned out to be 8 GB, not the 16 we
assumed, so every model now carries a "fits" flag; and the legal encoder we proposed was measured
against a simple baseline and lost — we'll show that on slide 8.

**Slide 4 — Progress against the plan**
Our plan has 27 milestones across six phases. Today 10 are working, 11 exist as a first version, and 6
are not started. The plan asked for a thin vertical slice by this point; we have a complete slice —
every stage is present and runs end to end.

▶ **Slide 5 — What runs end to end**
This is the pipeline as built, as one LangGraph graph. Green boxes work end to end, amber ones are a
working first version with a planned upgrade. Audio goes in on the left; a filled IF-1 draft comes out
on the right. Three properties hold everywhere: it runs fully offline on one laptop, every extracted fact
points back to where it came from in the transcript, and nothing in the system can mark a record as
verified — that is the officer's act.

**Slide 6 — Stage A, speech**
We use Whisper large-v3 running locally. On 60 public Tamil test clips, zero-shot, we measure word error
rate 51 percent and character error rate 15 percent. Two things to say about that number. First, Tamil
glues words together, so one boundary decision changes the word count — that's why we always show CER,
which is the honest metric. Second, that number was 57 and 22 percent last week and ran slower than real
time; we profiled the decoder, found the library was re-decoding uncertain audio up to six times at
increasing randomness, turned that off and switched to batched decoding — four times faster and more
accurate in one measured change.
We also added guards: on 2 of 60 clips Whisper invented repeated text; the guard catches it and the draft
is marked "not auto-resolvable". We then tried to fix the loop inside the decoder with a repetition
penalty — it cleared those two clips but made every other clip worse, so we rejected it and kept the
guard. That negative result is on slide 13.

**Slide 7 — Stage B, extraction, and the translation view**
Extraction is rule-based for now: dates, times, rupee amounts including spoken forms like "two lakh",
phone numbers and vehicle plates, each with character offsets. On our 28-document adversarial set it
gets 63 of 63 with zero false positives — and zero false positives is the property a form needs: a
blank prompts a question, a wrong value gets trusted.
The statute classifier reads English, so Tamil is translated *only* for the classifier — everything
else keeps the original. The interim translator scores chrF 35; the better model is gated behind a
login and is the planned upgrade.

▶ **Slide 8 — Stage C, the encoder question**
This is the research core. We trained on ILSI — 66,000 Indian court documents labelled with IPC sections
— and tested on 13,039 unseen documents. A TF-IDF logistic model reaches micro-F1 0.70. InLegalBERT,
the legal-domain encoder from our Review 2 plan, reaches 0.53 after every lever we could pull — more
epochs, a higher learning rate for its head, keeping both ends of long documents. Combining the two
adds 0.0006. We also ruled out truncation as the cause. So the conclusion for this corpus is settled:
the sparse model is our classifier, and the encoder's future job is verifying legal ingredients, not
picking sections.

**Slide 9 — The legal knowledge base**
Our training labels are IPC-era; police file under BNS 2023. A mapping table converts them — 69 of 100
rows are applied automatically, 29 are held for a human, because the source is a third-party directory,
not the gazette. For cognizability — whether an offence allows an FIR — we parsed the BNSS First Schedule
from the official gazette PDF: 438 entries, with page citations. That parse caught four errors in the
hand-written table we had been using, all in the direction of wrongly refusing an FIR. It also revealed
"conditional" entries the table didn't know: theft under five thousand rupees is non-cognizable, and
cruelty is cognizable only when the victim or a relative reports it. Those now route to the officer with
the Schedule's own words. All 358 BNS section texts are parsed too, so the officer sees the law, not a
number.

**Slide 10 — Element-wise justification and the safety net**
For 29 common offences we check the legal ingredients against the complaint — theft needs taking, property,
someone else's possession, no consent — and show which are evidenced, quoting the words. Never "no":
absence of a word is a question for the officer, not a verdict. The same cues run on the *original Tamil*
as a safety net, because we saw a real failure: the translator turned "dowry" into "relief" and a cruelty
complaint routed to the non-serious path. Now, if the ingredients of a serious offence are present in the
original text but the classifier missed it, the case goes to the officer with that section named.

▶ **Slide 11 — Stages D and E**
Stage D fills the fixed IF-1 deterministically — no model writes into structured fields — and composes a
short narrative where every sentence traces to an extracted fact; items 13 to 15 stay blank because they
are the officer's. Stage E is the officer page: type a complaint, upload audio, or record from the
microphone; it shows the route, the sections with their BNS text and ingredients, "also consider"
suggestions, flags, and the filled form. "Mark verified" is disabled by design, and every draft is written
to an append-only audit log — identifiers and hashes, never the text or the audio.
[Demo here or after slide 12 — see docs/REVIEW3_DEMO.md, six steps, about six minutes.]

**Slide 12 — Interim results at a glance**
One table: statute-ID 0.70 versus 0.53; speech 57 to 51 WER and 22 to 15 CER, four times faster;
translation chrF 35; extraction 63 of 63; the Schedule 438 entries; 376 automated tests. Every row
names its dataset and its file.

▶ **Slide 13 — Challenges and corrective action**
Each row is a problem we hit, the measurement that proved it, and what we did: the 8 GB GPU; the
encoder head collapsing to 6 of 100 labels until we weighted the loss; Whisper inventing text; the
translation losing "dowry"; the cognizability table being wrong in four places; the decoder running
three times slower than real time; and the repetition penalty we tried and rejected. The pattern we'd
like you to notice: every safety guard in the system exists because we reproduced a real failure first.

**Slide 14 — Technical accuracy and practice**
Thresholds tuned on the development split and reported on test; a baseline in every comparison; CER
beside WER; per-clip distributions, not just averages. The legal rules are tests: only high-confidence
mappings auto-apply, an unknown section never defaults, nothing can set "verified", a threat to kill is
intimidation, not attempted murder. The legal tables come from the gazette with page citations and are
awaiting a legal read-through — the form says so.

**Slide 15 — Report progress**
Chapters 1 to 3 are updated with a revisions section, and Chapter 4 — implementation, testing, results,
analysis, limitations — is drafted. [If asked: it is in the repository as Review3_Report.]
We acknowledge that implementation was carried out with the assistance of an AI coding assistant; all
design decisions, data choices, experiments and their interpretation were reviewed and are owned by the
team. [Confirm wording with the guide.]

**Slide 16 — Individual contribution**
[Fill honestly before the review. Each member speaks to their own row. The panel grades demonstrated
contribution; the repository log is the evidence.]

**Slide 17 — Plan to Review 4**
Four things, in order: a Tamil-fine-tuned speech model to bring the error rate down; in-domain synthetic
complaint data with a local LLM, which is the fix for the classifier on short complaints; the LLM element
verifier and narrative against the same schema; and the React verification workspace with accept and
override per section. Plus the legal read-through, and a publication venue agreed with our guide.

**Slide 18 — Summary**
We promised a thin slice; we have a full slice. Every stage runs, every number is measured on public
data, and the system is honest about what it doesn't know — it asks the officer. Thank you; we'll take
questions and show the live demo.

---

## If they ask (one breath each)

- *Why is the classifier empty on your own theft sample?* It's trained on long court text; on a two-line
  complaint its top five don't include theft, so lowering the threshold would apply wrong sections. The
  cue scan carries short inputs as a question; in-domain data is the fix.
- *Is 51% WER usable?* As decision support with provenance and guards, yes; as a transcript to file, no.
  It's zero-shot on read Wikipedia Tamil; a Tamil-fine-tuned model is the next step.
- *Why not the BERT you proposed?* We measured it: 0.53 versus 0.70, ensemble adds nothing.
- *Who verified the legal tables?* Nobody yet — parsed from the gazette with page citations, 25 spot
  checks in tests, awaiting a legal read-through.
- *Can it file an FIR?* No path exists; a test fails if one appears.
- *What was AI's role?* Coding assistance, acknowledged; decisions and interpretation are ours.
