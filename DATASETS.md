# DATASETS.md — Availability, Usability & Doability (verified)

> Verified live on 2026-08-17 by hitting Hugging Face's API, Zenodo, and GitHub directly — not from
> memory. Sizes/licenses/access below are real. Sample rows are saved in `data_samples/`.

## Verdict: **DOABLE.** ✅

Every pillar of the pipeline has real, openly-downloadable data. Two items are *build-your-own*
(expected, already in the plan), not blockers:

| Pillar | Data exists & open? | Verdict |
|--------|--------------------|---------|
| Tamil ASR | ✅ FLEURS-ta, Kathbath (1,684 h), Common Voice-ta, IndicSUPERB | **Green** — abundant |
| Statute identification | ✅ ILSI (~66K docs) on Zenodo, LeSICiN code on GitHub | **Green** — but IPC-era → needs BNS mapping |
| Legal NER (seed) | ✅ OpenNyAI InLegalNER (46,545 entities, 14 types) | **Green** |
| Statute text (RAG/KB) | ✅ BNS/BNSS from India Code + Kaggle BNS (358 sections) | **Green** |
| IF-1 slot extraction | ⚠️ No public Tamil complaint→IF-1 set | **Yellow** — synthesize (planned) |
| IPC→BNS mapping | ⚠️ MHA tables exist as PDFs | **Yellow** — build machine-readable table (planned) |

## Verified dataset registry

| Dataset | Role | Volume | License | Access | Format |
|---------|------|--------|---------|--------|--------|
| **google/fleurs** (`ta_in`) | ASR eval/train | **3,335 clips, ~2.76 GB** (train 2,367 / dev / test) — API-confirmed | CC-BY-4.0 | HF hub, open | parquet + audio |
| **ai4bharat/Kathbath** | ASR train | **1,684 h / 12 langs incl. Tamil** | CC0 / CC-BY-4.0 | HF hub, open | m4a + transcripts |
| **Common Voice (ta)** | ASR train/eval | ~hundreds of hrs (v17) | CC0 | HF hub (accept terms) | mp3 + tsv |
| **AI4Bharat IndicSUPERB** | ASR benchmark | multi-task Indic speech | CC-BY-4.0 | GitHub + HF | wav |
| **ILSI** (Zenodo `10.5281/zenodo.6053791`) | Statute ID | train 320 MB / dev 85 MB / test 106 MB `.jsonl`; 100 IPC sections | research use | **Zenodo direct, no form** | jsonl |
| **Law-AI/LeSICiN** | Statute-ID baseline code | — | MIT-style | GitHub | python |
| **opennyaiorg/InLegalNER** | Legal NER seed | 46,545 entities / 14 types | MIT | HF + GitHub (raw JSON) | spaCy JSON |
| **BNS/BNSS bare acts** | Statute text (RAG) | 358 sections / 20 chapters (BNS) | Govt. of India (public) | India Code / NCRB PDF / Kaggle | PDF → parse |
| **MHA IPC↔BNS tables** | Mapping layer | ~511 IPC → BNS rows | Govt. of India (public) | MHA PDFs | PDF → build table |

Access confirmations from this session: FLEURS size via `datasets-server/size`; Kathbath via HF
dataset card; ILSI file list via Zenodo API (`secs.jsonl`, `train/dev/test.jsonl`, `label_vocab.json`,
`type_map.json`, `citation_network.json`, `ils2v.bin`); InLegalNER via HF (note: not indexed by
datasets-server because it ships as raw spaCy-JSON — download the files directly, it is **not** gated).

## Real samples (saved in `data_samples/`)

**A statute entry** — `data_samples/ilsi_statutes_sample.jsonl`:
```json
{"id": "Section 2 in The Indian Penal Code", "text": ["Every person shall be liable to punishment under this Code ... within India."]}
```

**A fact/case instance** — `data_samples/ilsi_fact_sample.jsonl` (real dowry-death case, abridged):
```json
{"id": "100002997",
 "text": ["Tuliya Devi ... was married to Gullu 5 years ago.",
          "After marriage Tuliya Devi was treated with cruelty ... for demand of dowry.",
          "... committed murder of Tuliya Devi on 8.5.2010 ...", "..."],
 "labels": ["Section 304 in The Indian Penal Code", "Section 302 ...", "Section 498A ...",
            "Section 304B ...", "Section 300 ...", "Section 34 ..."]}
```

**Why the mapping layer matters** — those IPC labels map to BNS 2023 as:

| IPC (ILSI label) | Offence | → BNS 2023 |
|------------------|---------|-----------|
| 302 | Punishment for murder | **103** |
| 304 | Culpable homicide not amounting to murder | **105** |
| 304B | Dowry death | **80** |
| 498A | Cruelty by husband/relatives | **85** (86 defines cruelty) |
| 300 | Murder (definition) | **101** |
| 34 | Common intention | **3(5)** |

So: train statute-ID on ILSI's IPC labels → apply the mapping → emit BNS sections. The mapping table is
the one artifact we must build (from MHA correspondence PDFs) and human-verify.

## Turnkey download (dev machine)

> **This machine note:** Smart App Control blocks `pyarrow` here, and HF `datasets`/`bitsandbytes`
> depend on it. Do data-heavy + training work inside **WSL2 (Ubuntu) or a Docker container** for a
> clean Linux env, OR avoid `datasets` and read parquet with **DuckDB** + `huggingface_hub`. See
> IMPLEMENTATION.md §Environment. Everything stays local (no hosted APIs) either way.

```bash
# ILSI (statute ID) — direct from Zenodo, no form
for f in secs.jsonl label_vocab.json dev.jsonl test.jsonl train.jsonl type_map.json; do
  curl -L -o "data/ilsi/$f" "https://zenodo.org/records/6053791/files/$f?download=1"
done

# FLEURS Tamil + Kathbath (ASR) — via huggingface_hub (no pyarrow needed for snapshot)
huggingface-cli download google/fleurs --repo-type dataset --include "ta_in/*" --local-dir data/fleurs
huggingface-cli download ai4bharat/Kathbath --repo-type dataset --include "*tamil*" --local-dir data/kathbath

# Legal NER
git clone https://github.com/Legal-NLP-EkStep/legal_NER data/opennyai_ner
```

Approx dev disk footprint: ILSI ~0.5 GB, FLEURS-ta ~2.8 GB, Kathbath-ta a few GB, NER < 0.2 GB,
statute text < 50 MB → tens of GB, comfortable on a normal SSD (this is **disk**, not the 16 GB VRAM).

## Honest caveats (already tracked as risks)

1. **IF-1 extraction has no public Tamil corpus** → synthetic-first (PLAN §5.2), OpenNyAI NER seeds the
   entity types. This is the main "make our own data" effort.
2. **ILSI is IPC-era and English court-fact text**, not Tamil spoken complaints → (a) build the IPC→BNS
   mapping; (b) do statute reasoning on the extracted/translated English narrative (domain shift = risk
   R10, mitigated with in-domain synthetic + human gold).
3. **Common Voice / some HF sets require accepting terms** once (free HF account).
