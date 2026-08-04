# Data — provenance, licenses & fetching

Everything under `Data/` is either small project fixtures, Special Documents
samples, or the fresh benchmark corpora used by the Layer-1 curriculum and the
P24–P36 projects.

## Provenance

Two manifests record every downloaded file, one line per file
(`url|sha256|bytes|relpath`):

- `Data/.corpus-manifest.txt` — benchmark corpora under `Data/corpus/`
  (fetched by `scripts/fetch_fresh_corpus.py` and
  `scripts/fetch_gutenberg.py`)
- `Data/.samples-manifest.txt` — Special Documents samples under `Data/SD-0N-*`
  (fetched by `scripts/fetch_sd_samples.py`)

Re-downloading is only needed when the manifest cannot verify a file; the
fetchers are idempotent and skip files whose sha256 already matches.

## Corpora & licenses

| Corpus | Path | License | Source |
|---|---|---|---|
| Gutenberg | `Data/corpus/gutenberg/` | Public domain | Project Gutenberg |
| rag-mini-wikipedia | `Data/corpus/rag-mini-wikipedia/` | CC BY 4.0 | Hugging Face `rag-datasets/rag-mini-wikipedia` |
| BEIR fiqa / nfcorpus | `Data/corpus/beir-{fiqa,nfcorpus}/` | See below | UKP Darmstadt BEIR server |
| lost-in-the-middle | `Data/corpus/lost-in-the-middle/` | MIT (repo); NQ-derived data see below | `nelson-liu/lost-in-the-middle` (GitHub) |
| SciFact | `Data/corpus/scifact/` | CC BY-NC-SA 3.0 | official S3 release |
| HotpotQA | `Data/corpus/hotpotqa/` | CC BY-SA 4.0 | Wayback snapshot of the dead CMU host |
| MIND-small | fetched only with `--mind` | MSR research license — non-commercial research use only | Microsoft |

### Notes

- **Gutenberg** texts are public-domain works; `loaders/gutenberg.py` strips
  the Project Gutenberg boilerplate. This corpus is the primary data for
  Layer-1 track 01 (chunking).
- **BEIR** (`beir-cellar/beir`) is an Apache-2.0 *framework*, but it aggregates
  third-party datasets (FiQA-2018, NFCorpus, …) under their **own** licenses.
  The BEIR authors explicitly disclaim that you are licensed to use any
  dataset — check each original dataset's terms before redistribution.
- **lost-in-the-middle**: the repository is MIT-licensed; its `qa_data/` is
  derived from Google Natural Questions, which is released under CC BY-SA 3.0.
- **MIND-small** is gated behind `--mind` because it is non-commercial
  research use only.

## Local samples

`Data/local-docs/`, `Data/sample.csv`, `Data/sample.json` are small original
fixtures; the `Data/SD-0N-*` samples are public documents (public-domain
prose, SEC filings, sample invoices) used by the Special Documents series.
