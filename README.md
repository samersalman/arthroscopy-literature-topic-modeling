# BERTopic Analysis of Arthroscopy Literature

[![DOI](https://zenodo.org/badge/1279619889.svg)](https://doi.org/10.5281/zenodo.20836781)

> **Manuscript status:** the accompanying manuscript is currently *submitted for review*.

Automated bibliometric + NLP analysis of *Arthroscopy: Journal of Arthroscopic and Related Surgery* literature (1995–2025) using transformer-based topic modeling (BERTopic) with traditional bibliometric comparison (VOSviewer).

---

## What this study does

1. Retrieves all qualifying English abstracts from target sports medicine journals via PubMed
2. Embeds them with **MPNet** (`sentence-transformers/all-mpnet-base-v2`)
3. Discovers latent research topics using **BERTopic** (UMAP + HDBSCAN + c-TF-IDF)
4. Classifies each topic's trajectory: **Hot / Cold / Emerging / Stable** using linear regression with Benjamini-Hochberg FDR correction
5. Compares BERTopic topics against **VOSviewer** keyword co-occurrence clusters
6. Produces 7 interactive HTML figures and 5 publication-ready CSV tables

**Corpus:** *Arthroscopy: Journal of Arthroscopic and Related Surgery*, all qualifying abstracts from 1995–2025.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | On Python 3.9, use `requirements_py39.txt` instead |
| PubMed email | Register free at https://www.ncbi.nlm.nih.gov/account/ |
| ~20 GB disk space | For model weights, embeddings, and all outputs |
| 16 GB RAM recommended | UMAP + HDBSCAN on ~7k docs |
| GPU optional | CPU works; embedding takes ~60–90 min without GPU |

---

## Quick Start (5 steps)

```bash
# 1. Clone / download this folder and enter it
cd arthroscopy-literature-topic-modeling

# 2. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r files/requirements.txt

# 4. Add your PubMed email to .env
#    Open .env in any text editor and replace the placeholder:
#    PUBMED_EMAIL=your_email@institution.edu

# 5. Run the full pipeline
bash run_pipeline.sh
```

The pipeline will print progress for each phase and takes approximately **3–5 hours** end-to-end on a modern CPU (embedding is the bottleneck).

---

## Running phases individually

All scripts are run from the **project root** with the venv active.

```bash
# Phase 0: Verify all dependencies
python src/00_verify_setup.py

# Phase 1: Fetch abstracts from PubMed (~30–40 min without API key)
python src/01_data_collection/fetch_pubmed.py --journal ARTHROSCOPY

# Quality check raw data
python src/01_data_collection/check_raw_data.py \
    --input "data/raw/pubmed_raw_ARTHROSCOPY_*.csv"

# Phase 2: Preprocess corpus
python src/02_preprocessing/preprocess.py \
    --input "data/raw/pubmed_raw_ARTHROSCOPY_*.csv" \
    --journal ARTHROSCOPY

# Phase 2b: VOSviewer export (traditional bibliometric comparison)
python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY

# Phase 3: Generate embeddings (slow on CPU — go get coffee)
python src/03_modeling/embed.py \
    --input data/processed/corpus_clean_ARTHROSCOPY.csv \
    --model mpnet \
    --journal ARTHROSCOPY

# Phases 4-6: BERTopic with hyperparameter grid search
python src/03_modeling/run_bertopic.py \
    --corpus data/processed/corpus_clean_ARTHROSCOPY.csv \
    --embeddings data/embeddings/embeddings_mpnet_ARTHROSCOPY.npy \
    --journal ARTHROSCOPY \
    --grid_search

# Phase 7: Temporal trend analysis (with FDR correction)
python src/04_analysis/temporal_analysis.py --journal ARTHROSCOPY

# Phase 8: Validation metrics
python src/04_analysis/validate.py --journal ARTHROSCOPY

# Phase 9: Generate all figures
python src/05_visualization/generate_figures.py --journal ARTHROSCOPY

# Phase 10: Generate publication tables
python src/05_visualization/generate_tables.py --journal ARTHROSCOPY
```

---

## Project structure

```
arthroscopy-literature-topic-modeling/
├── README.md                     ← You are here
├── run_pipeline.sh               ← One-command full execution
├── .env                          ← API credentials (fill in PUBMED_EMAIL)
├── files/
│   ├── PROTOCOL.md               ← Full study protocol (read first)
│   ├── study_config.yaml         ← All hyperparameters documented
│   ├── requirements.txt          ← Pinned Python dependencies
│   └── requirements_py39.txt     ← Python 3.9 compatible versions
├── src/
│   ├── 00_verify_setup.py        ← Phase 0: dependency check
│   ├── 01_data_collection/
│   │   ├── fetch_pubmed.py       ← Phase 1: PubMed retrieval
│   │   └── check_raw_data.py     ← Data quality report
│   ├── 02_preprocessing/
│   │   └── preprocess.py         ← Phase 2: text cleaning
│   ├── 03_modeling/
│   │   ├── embed.py              ← Phase 3: MPNet embeddings
│   │   ├── run_bertopic.py       ← Phases 4-6: UMAP + HDBSCAN + BERTopic
│   │   └── label_topics_llm.py   ← Optional: GPT-4o topic labels
│   ├── 04_analysis/
│   │   ├── temporal_analysis.py  ← Phase 7: trend classification (BH-FDR)
│   │   └── validate.py           ← Phase 8: coherence + silhouette
│   ├── 05_visualization/
│   │   ├── generate_figures.py   ← Phase 9: 7 interactive HTML figures
│   │   └── generate_tables.py    ← Phase 10: 5 publication CSV tables
│   ├── 06_vosviewer/
│   │   └── export_for_vosviewer.py ← VOSviewer comparison export
│   └── utils/
│       └── evaluation.py         ← Shared coherence utilities
├── data/
│   ├── raw/                      ← PubMed CSVs + fetch log
│   ├── processed/                ← Cleaned corpus + preprocessing report
│   └── embeddings/               ← .npy embedding arrays + metadata JSON
└── outputs/
    ├── models/                   ← Saved BERTopic model + config JSON
    ├── figures/                  ← fig1–fig7 HTML files
    ├── tables/                   ← topic_info, doc_topics, temporal_trends, validation
    ├── reports/                  ← table1–table5 publication CSVs
    └── vosviewer/                ← RIS file, co-occurrence matrix, instructions
```

---

## Expected outputs

After a successful run you will have:

| File | Description |
|------|-------------|
| `data/raw/pubmed_raw_ARTHROSCOPY_*.csv` | ~17,800 raw abstracts |
| `data/processed/corpus_clean_ARTHROSCOPY.csv` | 7,166 cleaned documents (after publication-type + length filtering) |
| `data/embeddings/embeddings_mpnet_ARTHROSCOPY.npy` | Shape (N, 768) |
| `outputs/models/bertopic_model_ARTHROSCOPY/` | Loadable BERTopic model |
| `outputs/tables/grid_search_ARTHROSCOPY.csv` | 16 hyperparameter combinations |
| `outputs/tables/topic_info_ARTHROSCOPY.csv` | All topics with top words |
| `outputs/tables/temporal_trends_ARTHROSCOPY.csv` | Slope, p, p_adj, trend class |
| `outputs/tables/validation_report_ARTHROSCOPY.csv` | Coherence, silhouette, noise % |
| `outputs/figures/fig1_topic_map_ARTHROSCOPY.html` | Intertopic distance map |
| `outputs/figures/fig4_temporal_heatmap_ARTHROSCOPY.html` | Research activity heatmap |
| `outputs/reports/table2_all_topics_ARTHROSCOPY.csv` | Manuscript Table 2 |
| `outputs/vosviewer/vosviewer_ris_ARTHROSCOPY.ris` | VOSviewer import file |
| `data/raw/fetch_log_YYYYMMDD.txt` | Query log for reproducibility |

---

## Validation targets

| Metric | Target | Interpretation |
|--------|--------|----------------|
| C_NPMI coherence | > 0.15 | Good semantic quality |
| Silhouette score | > 0.30 | Moderate cluster separation |
| Noise ratio | < 25% | Most documents assigned to a topic |

A PubMedBERT sensitivity run is available via `--model pubmedbert`.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `PUBMED_EMAIL` error | Placeholder not replaced | Edit `.env` |
| `OOM` during embedding | Batch too large | Add `--batch_size 16` |
| Too many topics (>100) | `min_cluster_size` too small | Grid search will handle this |
| >40% noise | `min_samples` too high | Reduce in grid: add 5 to the list |
| UMAP >30 min | Large corpus, no GPU | Set `low_memory=True` in `build_topic_model()` in `run_bertopic.py` |
| `ModuleNotFoundError` | Wrong working directory | Always run from project root |
| `urllib3 OpenSSL warning` | macOS LibreSSL | Benign warning, ignore |

---

## Reproducibility

- All scripts use `random_state=42`
- All dependencies are pinned in `files/requirements.txt`
- PubMed query strings and retrieval dates are logged to `data/raw/fetch_log_*.txt`
- Grid search results are saved before the final model fit
- Model saved with `serialization="safetensors"` for long-term compatibility
- Running the default pipeline (MPNet embeddings) reproduces the manuscript's 64-topic solution (C_NPMI ≈ 0.18, topic diversity ≈ 0.72, ~17% noise).

---

## Data & code availability

All analysis code in this repository is released under the MIT License.

No study data are distributed in this repository. The corpus consists of publicly available PubMed metadata (abstracts and bibliographic fields), which is **not** redistributed here.

Anyone can regenerate every table and figure by running `run_pipeline.sh` against PubMed using the query and parameters defined in `files/study_config.yaml`.

The code is permanently archived on Zenodo under an MIT license. Please cite the **concept DOI** [`10.5281/zenodo.20836781`](https://doi.org/10.5281/zenodo.20836781), which always resolves to the latest version. Each tagged release also receives its own version-specific DOI on Zenodo.

Repository: https://github.com/samersalman/arthroscopy-literature-topic-modeling

---

## Citation

If you use this code, please cite it using the metadata in `CITATION.cff` (GitHub's **"Cite this repository"** button generates a formatted citation). The accompanying manuscript is currently **submitted for review**.

For questions about the study design, see `files/PROTOCOL.md` (protocol version 1.0).
