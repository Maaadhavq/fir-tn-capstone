# data/statutes

| file | what | provenance |
|------|------|-----------|
| `mha_250884_english.pdf` | The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023), English gazette text, 249 pp. | Gazette of India Extraordinary, Part II Sec. 1, No. 55, 25 Dec 2023 (CG-DL-E-25122023-250884); copy hosted by MHA at `mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf`, fetched 2026-09-13 |
| `mha_250883_english.pdf` | The Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023), 102 pp. | Gazette No. 53, same date; MHA copy `250883_english_01042024.pdf`, fetched 2026-09-13 |
| `bns_2023.jsonl` | All 358 BNS sections: marginal heading, chapter, full text, `operative_text` (illustrations stripped), gazette page | built by `scripts/build_bns_text.py`; loaded by `fir.statute.bns_text` so the officer sees the BNS words next to a suggestion (the IPC text from ILSI is shown as lineage) |
| `bnss_schedule1_rows.csv` | The First Schedule, Part I ("Offences under the BNS"), one line per printed row, text verbatim, with gazette page | built by `scripts/build_bnss_schedule.py` |
| `bnss_schedule1.csv` | One line per BNS section / sub-clause: cognizable, bailable, triable_by, condition, notes, source_ref | same script; loaded by `fir.statute.cognizability` |

## Status: parsed from the official text, **not legally signed off**

The parser is a coordinate parser over a six-column table with no ruling lines.
It was checked by: 0 empty cells, section numbers in Schedule order and unique,
every classification cell reading `Cognizable` / `Non-cognizable` / `According
as ...` / `Cognizable if ...`, and a 25-entry spot check against known law
(`tests/test_bnss_schedule.py`). A reviewer should still read
`bnss_schedule1_rows.csv` against the printed Schedule; the `gazette_page`
column is there for that.

## How the table is reduced

- `Cognizable.` -> `cognizable`; `Non-cognizable.` -> `non_cognizable`.
- Any other wording -> `conditional`, with the Schedule's words in `notes`.
  Routing treats it like unknown (officer review) and shows the words.
- A section printed as several unlabelled rows that disagree (theft 303(2):
  cognizable, but non-cognizable "where value of property is less than 5,000
  rupees") -> `conditional`. Where the split is on a fact the pipeline extracts,
  `condition` holds a machine-readable rule
  (`property_value_inr<5000=non_cognizable;else=cognizable`) that the loader
  applies when that fact is present. The build asserts the printed rows still
  say what the rule encodes.
- Base numbers the Schedule does not print (`103` from `103(1)`/`103(2)`) are
  derived by the loader only where every sub-clause agrees; otherwise the base
  is `unknown` and asks the officer which clause.

Rebuild: `make schedule` (or `bash scripts/run.sh schedule`) -- prints the
disagreements with the old hand-coded stub (4 substantive: BNS 126, 223, 296,
329 were cognizable in the Schedule, non-cognizable in the stub).
