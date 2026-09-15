# IPC → BNS mapping table

Converts old **Indian Penal Code (IPC 1860)** sections to new **Bharatiya Nyaya Sanhita (BNS 2023)**
sections. Needed because our statute-ID training data (ILSI) is IPC-labelled, but police now file FIRs
under BNS (in force 1 July 2024).

## Status: verified against a BNS-IPC directory — ~71/100 now high-confidence

Covers the **100 IPC sections used by the ILSI dataset** (`ilsi_ipc_labels.json`). Each row has a
`confidence` (high|med|low) and `needs_review` flag.

- **~71 rows `high`** — corroborated by the BNS-IPC pocket directory (rendered + read this session)
  and/or 2 web conversion tables. The common FIR offences are all here (302→103(1), 304B→80,
  379→303(2), 420→318(4), 498A→85, 354→74, 506→351(2), ...).
- **~29 rows med/low `needs_review=Y`** — not in the directory or still uncertain (procedural/rare
  sections, application clauses, a few sub-clauses). Do **not** treat these as final.

## ⚠️ Source honesty
The PDF here (`BNS-IPC_pocket_directory.pdf`) is a widely-circulated **third-party "BNS vs IPC Pocket
Directory"** (watermark "Ansari Sir"), which happened to be hosted at a Govt.-of-India S3 URL. It is a
strong, internally-consistent secondary source — **not the official gazette notification.** For any
real/official use, still verify high-value rows against the **official MHA notification** and get a
law-qualified reviewer to sign off (PLAN.md risks R3/R7).

The directory even contains one internal slip (IPC 339-342 restraint/confinement cluster); we kept
`341 → BNS 126(2)` per the statute's own structure and flagged it.

## Files
- `ipc_bns_map.csv` — the mapping (columns below).
- `ilsi_ipc_labels.json` — the exact 100 IPC sections ILSI uses (join key).
- `BNS-IPC_pocket_directory.pdf` — the 2-page source directory (was mislabelled "official"; corrected).

## Columns
`ipc_section, offence, bns_section, confidence (high|med|low), needs_review (Y|N), notes`

## Conflicts resolved this pass
- IPC 325 → **BNS 117(2)** (blogs disagreed: 117(2) vs 118(2)).
- IPC 326 → **BNS 118(2)** (blogs disagreed: 118(2) vs 118(3)).
- IPC 353 → **BNS 132** (one blog wrongly said 121).
- IPC 363 → **BNS 137** (blog said 139).
- Sub-clauses corrected: 365→140(3), 380→305(a), 436→326G, 437→327(1), 438→327(8), 450→332(B).

## How to use (code)
```python
import csv
def load_ipc_bns_map(path="data/mapping/ipc_bns_map.csv"):
    return {r["ipc_section"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
# ILSI label "Section 302 in The Indian Penal Code" -> ipc_section "302" -> row["bns_section"]
# In the pipeline, only auto-apply rows with confidence=="high"; route needs_review=="Y" to a human.
```

## To finish hardening (later)
1. Verify the ~29 `needs_review` rows against the official MHA gazette; fill 155/156/190/482 (`VERIFY`).
2. Legal reviewer signs off; version the file (DVC); record the official notification reference.
