# BERTopic Sports Medicine Orthopedic Surgery — Full Study Protocol

> **For anyone running this study:** This document is the single source of truth for executing this study.
> Read it top to bottom before writing any code. Each phase is self-contained and ordered.
> All file paths are relative to the project root (`bertopic_sports_med/`).
> Every script referenced here either exists in `src/` or must be created as specified.

---

## Table of Contents

1. [Study Overview](#1-study-overview)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Data Collection](#3-phase-1--data-collection)
4. [Phase 2 — Preprocessing](#4-phase-2--preprocessing)
5. [Phase 3 — Embedding](#5-phase-3--embedding)
6. [Phase 4 — Dimensionality Reduction](#6-phase-4--dimensionality-reduction)
7. [Phase 5 — Clustering](#7-phase-5--clustering)
8. [Phase 6 — Topic Modeling](#8-phase-6--topic-modeling)
9. [Phase 7 — Temporal Analysis](#9-phase-7--temporal-analysis)
10. [Phase 8 — Validation](#10-phase-8--validation)
11. [Phase 9 — Visualization](#11-phase-9--visualization)
12. [Phase 10 — Reporting](#12-phase-10--reporting)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Expected Outputs Checklist](#15-expected-outputs-checklist)

---

## 1. Study Overview

### Research Question
What are the latent research themes in sports medicine orthopedic surgery literature published 1995–2025 (31 calendar years), how have they evolved temporally, and how does transformer-based topic modeling (BERTopic) compare to traditional bibliometric methods in identifying these themes?

### Primary Objectives
1. Identify and label all major research topics in sports medicine orthopedic surgery journals (1995–2025)
2. Quantify temporal trends (hot, cold, emerging, stable topics) with FDR-corrected p-values (Benjamini-Hochberg)
3. Compare BERTopic results against VOSviewer keyword co-occurrence analysis using Jaccard similarity (J≥0.20) and visual inspection
4. Produce a publication-ready manuscript with reproducible code

> **Scope note:** This corpus is journal-based and captures literature from selected high-impact sports medicine orthopedic journals, not all sports medicine literature. Manuscript language should reflect this (e.g., "in the five leading sports medicine orthopedic journals" not "in the sports medicine literature").

### Study Design
- **Type:** Bibliometric + NLP analysis (no IRB required — published abstracts only)
- **Corpus:** Published abstracts from target sports medicine journals, 1995–2025
- **Primary method:** BERTopic with MPNet (all-mpnet-base-v2) embeddings
- **Comparison method:** VOSviewer keyword co-occurrence (traditional bibliometrics)
- **Unit of analysis:** Individual abstract (title + abstract concatenated)

### Target Journals (Priority Order)
| Priority | Journal | Abbreviation | PubMed ISSN |
|----------|---------|--------------|-------------|
| 1 | American Journal of Sports Medicine | AJSM | 0363-5465 |
| 2 | Arthroscopy: Journal of Arthroscopic & Related Surgery | Arthroscopy | 0749-8063 |
| 3 | Knee Surgery, Sports Traumatology, Arthroscopy | KSSTA | 0942-2056 |
| 4 | Journal of ISAKOS | JISAKOS | 2059-7754 |
| 5 | Orthopaedic Journal of Sports Medicine | OJSM | 2325-9671 |

**Recommended starting point:** AJSM only (Phase 1 target), then expand to Arthroscopy + KSSTA for a multi-journal analysis.

### Timeline
| Phase | Task | Estimated Duration |
|-------|------|--------------------|
| 1 | Data collection | Days 1–3 |
| 2 | Preprocessing | Days 3–4 |
| 3 | Embedding | Days 4–6 |
| 4–6 | Modeling | Days 6–8 |
| 7 | Temporal analysis | Days 8–10 |
| 8 | Validation | Days 10–14 |
| 9 | Visualization | Days 14–16 |
| 10 | Reporting | Days 16–30 |

---

## 2. Environment Setup

### 2.1 Python Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** (pinned versions for reproducibility; Python ≥3.10 recommended; on Python 3.9 use `scipy==1.13.1`):

> **`openai` is optional** — only needed for LLM topic labeling (Phase 6 optional step). Install separately with `pip install openai==1.40.0` if needed; it is commented out in requirements.txt.

```
# Core NLP / Topic Modeling
bertopic==0.16.4
sentence-transformers==3.0.1
transformers==4.44.2
torch==2.4.0

# Dimensionality Reduction & Clustering
umap-learn==0.5.6
hdbscan==0.8.38.post1
scikit-learn==1.5.1

# Data Handling
pandas==2.2.2
numpy==1.26.4
pyarrow==17.0.0

# PubMed API
biopython==1.84
requests==2.32.3
ratelimit==2.2.1

# NLP Utilities
nltk==3.9.1
spacy==3.7.6

# Visualization
plotly==5.23.0
matplotlib==3.9.2
seaborn==0.13.2
pyvis==0.3.2

# Topic Coherence Evaluation
gensim==4.3.3

# Statistical Analysis
scipy==1.14.0
statsmodels==0.14.3
pingouin==0.5.4

# Reporting
jinja2==3.1.4
openpyxl==3.1.5
xlsxwriter==3.2.0

# Utilities
tqdm==4.66.5
python-dotenv==1.0.1
loguru==0.7.2
pyyaml==6.0.2
joblib==1.4.2
```

### 2.3 Download NLTK and spaCy Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

### 2.4 Environment Variables

Create `.env` in project root:

```bash
# .env — DO NOT COMMIT TO VERSION CONTROL
PUBMED_EMAIL=your_email@institution.edu   # Required by NCBI
PUBMED_API_KEY=your_key_here              # Optional but increases rate limit to 10 req/s
OPENAI_API_KEY=sk-...                     # Optional: only needed for LLM topic labeling
```

Load in all scripts with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 2.5 Verify Installation

```bash
python src/00_verify_setup.py
```

**Create `src/00_verify_setup.py`:**

```python
"""Verify all dependencies are correctly installed before running the pipeline."""
import sys

def check_import(module, name=None):
    try:
        __import__(module)
        print(f"  ✓ {name or module}")
        return True
    except ImportError as e:
        print(f"  ✗ {name or module}: {e}")
        return False

checks = [
    ("bertopic", "BERTopic"),
    ("sentence_transformers", "SentenceTransformers"),
    ("umap", "UMAP"),
    ("hdbscan", "HDBSCAN"),
    ("Bio.Entrez", "Biopython/Entrez"),
    ("pandas", "Pandas"),
    ("plotly", "Plotly"),
    ("gensim", "Gensim"),
    ("sklearn", "Scikit-learn"),
    ("torch", "PyTorch"),
]

print("Checking dependencies...")
results = [check_import(m, n) for m, n in checks]

if all(results):
    print("\n✓ All dependencies installed correctly. Ready to run pipeline.")
else:
    print(f"\n✗ {results.count(False)} dependency/dependencies failed. Run: pip install -r requirements.txt")
    sys.exit(1)
```

---

## 3. Phase 1 — Data Collection

### 3.1 Overview
Retrieve all qualifying abstracts from target journals via PubMed's Entrez API.
Output: `data/raw/pubmed_raw_{journal}_{date}.csv`

### 3.2 PubMed Query Construction

For each journal, use journal title **and** ISSN to prevent missed records due to PubMed abbreviation variations:

```
# AJSM
("Am J Sports Med"[Journal] OR "0363-5465"[ISSN]) AND ("1995/01/01"[PDAT] : "2025/12/31"[PDAT])
AND hasabstract[text] AND English[lang]

# Arthroscopy
("Arthroscopy"[Journal] OR "0749-8063"[ISSN]) AND ("1995/01/01"[PDAT] : "2025/12/31"[PDAT])
AND hasabstract[text] AND English[lang]

# KSSTA
("Knee Surg Sports Traumatol Arthrosc"[Journal] OR "0942-2056"[ISSN]) AND ("1995/01/01"[PDAT] : "2025/12/31"[PDAT])
AND hasabstract[text] AND English[lang]
```

All queries and retrieval dates are written to `data/raw/fetch_log_YYYYMMDD.txt` automatically.

### 3.3 Data Collection Script

**Create `src/01_data_collection/fetch_pubmed.py`:**

```python
"""
Phase 1: PubMed Abstract Retrieval
===================================
Fetches all qualifying abstracts from target journals via NCBI Entrez API.

Usage:
    python src/01_data_collection/fetch_pubmed.py --journal AJSM
    python src/01_data_collection/fetch_pubmed.py --journal ALL
    python src/01_data_collection/fetch_pubmed.py --journal AJSM --start_year 2000 --end_year 2025

Outputs:
    data/raw/pubmed_raw_AJSM_YYYYMMDD.csv
    data/raw/fetch_log_YYYYMMDD.txt
"""

import os
import time
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

Entrez.email = os.getenv("PUBMED_EMAIL", "researcher@example.com")
Entrez.api_key = os.getenv("PUBMED_API_KEY", "")  # Optional

JOURNAL_QUERIES = {
    "AJSM": '"Am J Sports Med"[Journal]',
    "ARTHROSCOPY": '"Arthroscopy"[Journal]',
    "KSSTA": '"Knee Surg Sports Traumatol Arthrosc"[Journal]',
    "JISAKOS": '"J ISAKOS"[Journal]',
    "OJSM": '"Orthop J Sports Med"[Journal]',
}

FIELDS_TO_EXTRACT = [
    "pmid", "title", "abstract", "year", "month",
    "journal", "journal_abbrev", "volume", "issue", "pages",
    "doi", "authors", "keywords", "pub_types", "language",
    "affiliation", "country",
]

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper Functions ───────────────────────────────────────────────────────────

def build_query(journal_key: str, start_year: int, end_year: int) -> str:
    """Construct PubMed search query for a journal and date range."""
    journal_term = JOURNAL_QUERIES[journal_key]
    date_term = f'("{start_year}/01/01"[PDAT] : "{end_year}/12/31"[PDAT])'
    return f'{journal_term} AND {date_term} AND hasabstract[text] AND English[lang]'


def search_pubmed(query: str, retmax: int = 100000) -> list[str]:
    """Return list of PMIDs matching the query."""
    logger.info(f"Searching PubMed: {query}")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    logger.info(f"Found {len(pmids)} records")
    return pmids


def fetch_records_batch(pmids: list[str], batch_size: int = 200) -> list[dict]:
    """Fetch full records for a list of PMIDs in batches."""
    records = []
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching batches"):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        try:
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
            batch_records = Entrez.read(handle)
            handle.close()
            for article in batch_records["PubmedArticle"]:
                parsed = parse_article(article)
                if parsed:
                    records.append(parsed)
        except Exception as e:
            logger.error(f"Batch {i//batch_size} failed: {e}")
            time.sleep(5)
            continue
        # Rate limiting: 10 req/s with API key, 3 req/s without
        sleep_time = 0.11 if Entrez.api_key else 0.34
        time.sleep(sleep_time)
    return records


def parse_article(article: dict) -> dict | None:
    """Extract relevant fields from a PubMed XML record."""
    try:
        medline = article["MedlineCitation"]
        art = medline["Article"]

        # Title
        title = str(art.get("ArticleTitle", "")).strip()

        # Abstract — handle structured abstracts
        abstract_text = ""
        if "Abstract" in art:
            abstract_obj = art["Abstract"].get("AbstractText", "")
            if isinstance(abstract_obj, list):
                # Structured abstract: concatenate labeled sections
                abstract_text = " ".join([
                    f"{getattr(sec, 'attributes', {}).get('Label', '')}: {str(sec)}"
                    for sec in abstract_obj
                ]).strip()
            else:
                abstract_text = str(abstract_obj).strip()

        if not abstract_text:
            return None  # Skip records without abstracts

        # Publication date
        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = str(pub_date.get("Year", "")).strip()
        month = str(pub_date.get("Month", "")).strip()
        if not year:
            year = str(medline.get("DateCompleted", {}).get("Year", "")).strip()

        # Journal info
        journal_info = art.get("Journal", {})
        journal_name = str(journal_info.get("Title", "")).strip()
        journal_abbrev = str(journal_info.get("ISOAbbreviation", "")).strip()
        volume = str(journal_info.get("JournalIssue", {}).get("Volume", "")).strip()
        issue = str(journal_info.get("JournalIssue", {}).get("Issue", "")).strip()

        # Pagination
        pages = str(art.get("Pagination", {}).get("MedlinePgn", "")).strip()

        # DOI
        doi = ""
        for id_obj in article.get("PubmedData", {}).get("ArticleIdList", []):
            if getattr(id_obj, "attributes", {}).get("IdType") == "doi":
                doi = str(id_obj).strip()
                break

        # Authors
        author_list = art.get("AuthorList", [])
        authors = []
        for author in author_list:
            last = str(author.get("LastName", "")).strip()
            fore = str(author.get("ForeName", "")).strip()
            if last:
                authors.append(f"{last} {fore}".strip())
        authors_str = "; ".join(authors)

        # Affiliation (first author)
        affiliation = ""
        if author_list:
            aff_list = author_list[0].get("AffiliationInfo", [])
            if aff_list:
                affiliation = str(aff_list[0].get("Affiliation", "")).strip()

        # Keywords
        keyword_list = medline.get("KeywordList", [[]])
        keywords = "; ".join([str(kw) for sublist in keyword_list for kw in sublist])

        # MeSH terms
        mesh_list = medline.get("MeshHeadingList", [])
        mesh_terms = "; ".join([
            str(mh.get("DescriptorName", "")) for mh in mesh_list
        ])

        # Publication types
        pub_types = "; ".join([
            str(pt) for pt in art.get("PublicationTypeList", [])
        ])

        # PMID
        pmid = str(medline["PMID"])

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "year": year,
            "month": month,
            "journal": journal_name,
            "journal_abbrev": journal_abbrev,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": doi,
            "authors": authors_str,
            "keywords": keywords,
            "mesh_terms": mesh_terms,
            "pub_types": pub_types,
            "affiliation": affiliation,
            "n_authors": len(authors),
        }
    except Exception as e:
        logger.warning(f"Failed to parse article: {e}")
        return None


def fetch_journal(
    journal_key: str,
    start_year: int = 1995,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Full pipeline: search → fetch → parse → save for one journal."""
    query = build_query(journal_key, start_year, end_year)
    pmids = search_pubmed(query)

    if not pmids:
        logger.warning(f"No PMIDs found for {journal_key}")
        return pd.DataFrame()

    records = fetch_records_batch(pmids)
    df = pd.DataFrame(records)

    if df.empty:
        logger.warning(f"No records parsed for {journal_key}")
        return df

    # Deduplicate by PMID
    df = df.drop_duplicates(subset="pmid")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "abstract"])
    df = df[df["year"] >= start_year]
    df = df.sort_values("year")

    # Save
    date_str = datetime.now().strftime("%Y%m%d")
    outpath = OUTPUT_DIR / f"pubmed_raw_{journal_key}_{date_str}.csv"
    df.to_csv(outpath, index=False)
    logger.success(f"Saved {len(df)} records to {outpath}")

    # Summary log
    logger.info(f"\n{'='*50}")
    logger.info(f"Journal: {journal_key}")
    logger.info(f"Total records: {len(df)}")
    logger.info(f"Year range: {df['year'].min():.0f}–{df['year'].max():.0f}")
    logger.info(f"Missing abstracts removed: {len(pmids) - len(df)}")
    logger.info(f"{'='*50}\n")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for sports medicine study")
    parser.add_argument("--journal", default="AJSM",
                        choices=list(JOURNAL_QUERIES.keys()) + ["ALL"],
                        help="Journal to fetch (default: AJSM)")
    parser.add_argument("--start_year", type=int, default=1995)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()

    journals = list(JOURNAL_QUERIES.keys()) if args.journal == "ALL" else [args.journal]

    all_dfs = []
    for j in journals:
        df = fetch_journal(j, args.start_year, args.end_year)
        all_dfs.append(df)

    if args.journal == "ALL" and all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset="pmid")
        date_str = datetime.now().strftime("%Y%m%d")
        combined.to_csv(OUTPUT_DIR / f"pubmed_raw_ALL_{date_str}.csv", index=False)
        logger.success(f"Combined dataset: {len(combined)} unique records")
```

### 3.4 Data Collection Execution

```bash
# Single journal (recommended start)
python src/01_data_collection/fetch_pubmed.py --journal AJSM

# All journals
python src/01_data_collection/fetch_pubmed.py --journal ALL

# Custom date range
python src/01_data_collection/fetch_pubmed.py --journal AJSM --start_year 2000 --end_year 2025
```

### 3.5 Expected Output

```
data/raw/
├── pubmed_raw_AJSM_20250101.csv       # ~15,000–20,000 rows
├── pubmed_raw_ARTHROSCOPY_20250101.csv
└── fetch_log_20250101.txt
```

**Required columns in raw CSV:**
`pmid, title, abstract, year, journal, doi, authors, keywords, mesh_terms, pub_types`

### 3.6 Data Quality Checks

After collection, run:
```bash
python src/01_data_collection/check_raw_data.py --input data/raw/pubmed_raw_AJSM_*.csv
```

**Create `src/01_data_collection/check_raw_data.py`:**

```python
"""Quick quality checks on the raw data before preprocessing."""
import argparse
import glob
import pandas as pd
from loguru import logger

def check_raw(path: str):
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    print(f"Total records:        {len(df)}")
    print(f"Unique PMIDs:         {df['pmid'].nunique()}")
    print(f"Missing abstracts:    {df['abstract'].isna().sum()}")
    print(f"Missing years:        {df['year'].isna().sum()}")
    print(f"Missing titles:       {df['title'].isna().sum()}")
    print(f"Year range:           {df['year'].min():.0f} – {df['year'].max():.0f}")
    print(f"Median abstract len:  {df['abstract'].dropna().str.len().median():.0f} chars")
    print(f"\nRecords per decade:")
    df['decade'] = (df['year'] // 10 * 10).astype(int)
    print(df.groupby('decade').size().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path or glob to raw CSV(s)")
    args = parser.parse_args()
    for path in sorted(glob.glob(args.input)):
        check_raw(path)
```

---

## 4. Phase 2 — Preprocessing

### 4.1 Overview
Clean and standardize abstracts for embedding. Output: `data/processed/corpus_clean_{journal}.csv`

### 4.2 Preprocessing Script

**Create `src/02_preprocessing/preprocess.py`:**

```python
"""
Phase 2: Text Preprocessing
============================
Cleans raw PubMed abstracts for BERTopic embedding.

Key decisions:
- DO NOT remove stopwords (BERT handles this via attention)
- DO remove structured abstract labels ("BACKGROUND:", "METHODS:", etc.)
- DO concatenate title + abstract (increases signal)
- DO filter by abstract length (min 50 words)
- DO NOT stem or lemmatize (BERT uses full subword tokenization)

Usage:
    python src/02_preprocessing/preprocess.py --input data/raw/pubmed_raw_AJSM_*.csv
    python src/02_preprocessing/preprocess.py --input data/raw/pubmed_raw_AJSM_*.csv --journal AJSM

Outputs:
    data/processed/corpus_clean_AJSM.csv
    data/processed/preprocessing_report_AJSM.txt
"""

import re
import argparse
import glob
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_ABSTRACT_WORDS = 50
MIN_ABSTRACT_CHARS = 200

# Structured abstract section labels to strip
STRUCTURED_LABELS = re.compile(
    r"\b(BACKGROUND|OBJECTIVE[S]?|PURPOSE|AIM[S]?|HYPOTHESIS|STUDY DESIGN|"
    r"METHODS?|MATERIALS? AND METHODS?|PATIENTS? AND METHODS?|DESIGN|"
    r"SETTING|PARTICIPANTS?|INTERVENTIONS?|RESULTS?|FINDINGS?|"
    r"CONCLUSIONS?|SIGNIFICANCE|CLINICAL RELEVANCE|LEVEL OF EVIDENCE|"
    r"EVIDENCE|WHAT IS KNOWN|WHAT THIS STUDY ADDS|TAKE[\-\s]HOME MESSAGE|"
    r"KEY POINTS?|TRIAL REGISTRATION)\s*:?\s*",
    re.IGNORECASE,
)

# Common boilerplate phrases to remove
BOILERPLATE = [
    r"Copyright ©.*?\.",
    r"All rights reserved\.",
    r"Published by.*?Elsevier.*?\.",
    r"This is an open[- ]access article.*?\.",
    r"ClinicalTrials\.gov.*?NCT\d+",
    r"Level of evidence:.*?\.",
    r"Study design:.*?\.",
]

# Publication types to EXCLUDE
EXCLUDE_PUB_TYPES = {
    "Editorial",
    "Letter",
    "Comment",
    "Published Erratum",
    "Retracted Publication",
    "News",
    "Biography",
    "Personal Narrative",
}

# ── Cleaning Functions ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply all text cleaning steps to an abstract or title."""
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text)

    # Remove structured abstract labels
    text = STRUCTURED_LABELS.sub(" ", text)

    # Remove boilerplate
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove HTML entities and tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing punctuation artifacts
    text = text.strip(".,;:-")

    return text.strip()


def build_document(row: pd.Series) -> str:
    """Concatenate title + abstract into a single document string."""
    title = clean_text(str(row.get("title", "")))
    abstract = clean_text(str(row.get("abstract", "")))

    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def is_valid_document(doc: str, min_words: int = MIN_ABSTRACT_WORDS) -> bool:
    """Check if a document meets minimum length requirements."""
    if not doc:
        return False
    words = doc.split()
    return len(words) >= min_words and len(doc) >= MIN_ABSTRACT_CHARS


def filter_pub_types(pub_types_str: str) -> bool:
    """Return True if publication type should be INCLUDED."""
    if pd.isna(pub_types_str):
        return True
    for exclude_type in EXCLUDE_PUB_TYPES:
        if exclude_type.lower() in str(pub_types_str).lower():
            return False
    return True


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def preprocess(input_glob: str, journal_label: str = None) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found matching: {input_glob}")

    logger.info(f"Loading {len(files)} file(s)...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    n_raw = len(df)
    logger.info(f"Loaded {n_raw} raw records")

    # ── Step 1: Remove duplicates ──────────────────────────────────────────────
    df = df.drop_duplicates(subset="pmid")
    n_after_dedup = len(df)
    logger.info(f"After dedup: {n_after_dedup} records ({n_raw - n_after_dedup} removed)")

    # ── Step 2: Filter publication types ──────────────────────────────────────
    df["include"] = df["pub_types"].apply(filter_pub_types)
    df = df[df["include"]].drop(columns=["include"])
    n_after_pubtype = len(df)
    logger.info(f"After pub type filter: {n_after_pubtype} records")

    # ── Step 3: Year validation ────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]
    n_after_year = len(df)
    logger.info(f"After year filter: {n_after_year} records")

    # ── Step 4: Build document strings ────────────────────────────────────────
    logger.info("Building documents (title + abstract)...")
    tqdm.pandas(desc="Building docs")
    df["document"] = df.progress_apply(build_document, axis=1)

    # ── Step 5: Filter short documents ────────────────────────────────────────
    df["is_valid"] = df["document"].apply(is_valid_document)
    df_invalid = df[~df["is_valid"]]
    df = df[df["is_valid"]].drop(columns=["is_valid"])
    n_final = len(df)
    logger.info(f"Removed {len(df_invalid)} short/empty documents")
    logger.info(f"Final corpus: {n_final} documents")

    # ── Step 6: Add derived columns ───────────────────────────────────────────
    df["doc_length_words"] = df["document"].str.split().str.len()
    df["doc_length_chars"] = df["document"].str.len()
    df["decade"] = (df["year"] // 10 * 10).astype(int)

    # ── Step 7: Sort ──────────────────────────────────────────────────────────
    df = df.sort_values("year").reset_index(drop=True)
    df["doc_id"] = df.index  # Stable index for all downstream joins

    # ── Save ───────────────────────────────────────────────────────────────────
    label = journal_label or "corpus"
    out_path = OUTPUT_DIR / f"corpus_clean_{label}.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved clean corpus: {out_path}")

    # ── Report ─────────────────────────────────────────────────────────────────
    report = f"""
PREPROCESSING REPORT — {label}
{'='*50}
Raw records loaded:        {n_raw}
After deduplication:       {n_after_dedup}
After pub type filter:     {n_after_pubtype}
After year filter:         {n_after_year}
After length filter:       {n_final}
Retention rate:            {n_final/n_raw*100:.1f}%

Year range:                {df['year'].min()} – {df['year'].max()}
Median doc length (words): {df['doc_length_words'].median():.0f}
Mean doc length (words):   {df['doc_length_words'].mean():.0f}

Records per year (last 5):
{df[df['year'] >= df['year'].max()-4].groupby('year').size().to_string()}
"""
    print(report)
    report_path = OUTPUT_DIR / f"preprocessing_report_{label}.txt"
    report_path.write_text(report)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PubMed abstracts")
    parser.add_argument("--input", required=True, help="Glob to raw CSV(s)")
    parser.add_argument("--journal", default="AJSM", help="Label for output files")
    args = parser.parse_args()
    preprocess(args.input, args.journal)
```

### 4.3 Preprocessing Execution

```bash
python src/02_preprocessing/preprocess.py \
    --input "data/raw/pubmed_raw_AJSM_*.csv" \
    --journal AJSM
```

### 4.4 Expected Output Schema

`data/processed/corpus_clean_AJSM.csv`:

| Column | Type | Description |
|--------|------|-------------|
| doc_id | int | Stable document ID (index) |
| pmid | str | PubMed ID |
| title | str | Original title |
| abstract | str | Original abstract |
| document | str | Cleaned title + abstract (used for embedding) |
| year | int | Publication year |
| decade | int | Publication decade |
| journal | str | Full journal name |
| doi | str | DOI |
| authors | str | Semicolon-separated author list |
| keywords | str | Author/MeSH keywords |
| doc_length_words | int | Word count of document |

---

## 5. Phase 3 — Embedding

### 5.1 Overview
Convert documents into dense semantic vectors using a pre-trained biomedical language model.
Output: `data/embeddings/embeddings_{model}_{journal}.npy`

### 5.2 Model Selection

| Model | Hugging Face ID | Notes |
|-------|----------------|-------|
| **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | Trained on PubMed abstracts; run as a domain-specific sensitivity comparison |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` | Best general-purpose sentence embedding model; reported model for this study |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | Fastest; use for >50k docs or limited VRAM |
| BiomedBERT-large | `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract` | Higher capacity; slower |

**Sensitivity analysis:** The manuscript reports all-mpnet-base-v2 as the primary embedding model; PubMedBERT is run as a domain-specific sensitivity comparison, with topic-level differences reported.

### 5.3 Embedding Script

**Create `src/03_modeling/embed.py`:**

```python
"""
Phase 3: Document Embedding
============================
Converts clean documents to dense semantic embeddings using sentence transformers.

Usage:
    python src/03_modeling/embed.py --input data/processed/corpus_clean_AJSM.csv
    python src/03_modeling/embed.py --input data/processed/corpus_clean_AJSM.csv \
        --model all-mpnet-base-v2 --batch_size 64

Outputs:
    data/embeddings/embeddings_pubmedbert_AJSM.npy
    data/embeddings/embedding_metadata_AJSM.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from loguru import logger

OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "biomedbert-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
}


def embed_documents(
    docs: list[str],
    model_key: str = "pubmedbert",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a list of documents using a sentence transformer model."""
    model_id = MODEL_MAP.get(model_key, model_key)
    logger.info(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id)

    logger.info(f"Embedding {len(docs)} documents (batch_size={batch_size})...")
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    logger.info(f"Embedding shape: {embeddings.shape}")
    return embeddings


def main(args):
    # Load corpus
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} documents from {args.input}")

    docs = df["document"].tolist()
    journal = args.journal or Path(args.input).stem.replace("corpus_clean_", "")

    # Embed
    embeddings = embed_documents(docs, args.model, args.batch_size)

    # Save embeddings
    model_short = args.model.replace("/", "_")
    emb_path = OUTPUT_DIR / f"embeddings_{model_short}_{journal}.npy"
    np.save(emb_path, embeddings)
    logger.success(f"Saved embeddings: {emb_path}")

    # Save metadata
    meta = {
        "model": args.model,
        "model_id": MODEL_MAP.get(args.model, args.model),
        "journal": journal,
        "n_documents": len(docs),
        "embedding_dim": embeddings.shape[1],
        "normalized": True,
        "created_at": datetime.now().isoformat(),
        "input_file": str(args.input),
    }
    meta_path = OUTPUT_DIR / f"embedding_metadata_{model_short}_{journal}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents for BERTopic")
    parser.add_argument("--input", required=True, help="Path to clean corpus CSV")
    parser.add_argument("--model", default="pubmedbert",
                        choices=list(MODEL_MAP.keys()), help="Embedding model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Embedding batch size (reduce if OOM)")
    parser.add_argument("--journal", default=None, help="Label for output files")
    args = parser.parse_args()
    main(args)
```

### 5.4 Embedding Execution

```bash
# Primary model
python src/03_modeling/embed.py \
    --input data/processed/corpus_clean_AJSM.csv \
    --model mpnet \
    --batch_size 32

# Sensitivity comparison model
python src/03_modeling/embed.py \
    --input data/processed/corpus_clean_AJSM.csv \
    --model pubmedbert \
    --batch_size 64
```

**Hardware notes:**
- GPU recommended (CUDA): ~5–10 min for 20k docs with PubMedBERT
- CPU fallback: ~60–90 min for 20k docs (still functional)
- Memory: ~4GB RAM minimum; 8GB+ recommended

---

## 6. Phase 4 — Dimensionality Reduction

### 6.1 UMAP Configuration

UMAP reduces 768-dimensional embeddings to a low-dimensional space for clustering.

```python
from umap import UMAP

umap_model = UMAP(
    n_neighbors=15,        # Controls local vs global structure
                           # Lower = more local (more small clusters)
                           # Higher = more global (fewer large clusters)
                           # Range to test: [10, 15, 20, 30]
    n_components=5,        # Intermediate dims for clustering (NOT visualization)
                           # Keep at 5 per BERTopic recommendation
    min_dist=0.0,          # Minimum distance between embedded points
                           # 0.0 = tighter clusters (recommended for clustering)
    metric="cosine",       # Best for normalized sentence embeddings
    random_state=42,       # For reproducibility — ALWAYS set this
    low_memory=False,      # Set True if RAM is limited (>50k docs)
)
```

**Sensitivity analysis — UMAP grid:**

| Parameter | Values to Test |
|-----------|---------------|
| n_neighbors | 10, 15, 20, 30 |
| n_components | 5 (fixed) |
| metric | cosine (fixed) |

Run all combinations; report number of topics discovered per configuration.

---

## 7. Phase 5 — Clustering

### 7.1 HDBSCAN Configuration

```python
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(
    min_cluster_size=50,       # Minimum documents per topic
                                # Rule of thumb: ~0.2-0.5% of corpus
                                # For 20k docs: try 40, 50, 75, 100
    min_samples=10,             # Conservativeness of clustering
                                # Higher = more noise, fewer topics
                                # Try: 5, 10, 15
    metric="euclidean",         # Use after UMAP (UMAP output is Euclidean)
    cluster_selection_method="eom",  # "eom" or "leaf"
                                     # "eom" = more balanced cluster sizes
    prediction_data=True,       # REQUIRED for soft clustering / probabilities
    core_dist_n_jobs=-1,        # Use all CPU cores
)
```

**Sensitivity analysis — HDBSCAN grid:**

| Parameter | Values to Test |
|-----------|---------------|
| min_cluster_size | 30, 50, 75, 100, 150 |
| min_samples | 5, 10, 15 |
| cluster_selection_method | "eom", "leaf" |

---

## 8. Phase 6 — Topic Modeling (Full Pipeline)

### 8.1 Full BERTopic Pipeline Script

**Create `src/03_modeling/run_bertopic.py`:**

```python
"""
Phase 6: Full BERTopic Pipeline
================================
Runs the complete BERTopic pipeline with hyperparameter tuning.

Usage:
    # Standard run
    python src/03_modeling/run_bertopic.py \
        --corpus data/processed/corpus_clean_AJSM.csv \
        --embeddings data/embeddings/embeddings_pubmedbert_AJSM.npy \
        --journal AJSM

    # With hyperparameter grid search
    python src/03_modeling/run_bertopic.py \
        --corpus data/processed/corpus_clean_AJSM.csv \
        --embeddings data/embeddings/embeddings_pubmedbert_AJSM.npy \
        --journal AJSM \
        --grid_search

Outputs:
    outputs/models/bertopic_model_AJSM/     (saved BERTopic model)
    outputs/tables/topic_info_AJSM.csv      (topic summary table)
    outputs/tables/doc_topics_AJSM.csv      (per-document topic assignments)
    outputs/tables/grid_search_AJSM.csv     (if --grid_search)
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from loguru import logger
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [MODEL_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Coherence Evaluation ───────────────────────────────────────────────────────

def compute_coherence(topic_model: BERTopic, docs: list[str], top_n: int = 10) -> dict:
    """Compute NPMI topic coherence using gensim."""
    try:
        topics = topic_model.get_topics()
        # Build topic word lists (exclude noise topic -1)
        topic_words = [
            [word for word, _ in topics[t_id][:top_n]]
            for t_id in sorted(topics.keys())
            if t_id != -1
        ]

        # Tokenize docs
        tokenized = [doc.lower().split() for doc in docs]
        dictionary = Dictionary(tokenized)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized]

        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        score = coherence_model.get_coherence()
        return {"coherence_npmi": round(score, 4), "n_topics": len(topic_words)}
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        return {"coherence_npmi": None, "n_topics": len(topic_model.get_topics()) - 1}


def compute_diversity(topic_model: BERTopic, top_n: int = 10) -> float:
    """Compute topic diversity (proportion of unique words across all topics)."""
    try:
        topics = topic_model.get_topics()
        all_words = [
            word
            for t_id, words in topics.items()
            if t_id != -1
            for word, _ in words[:top_n]
        ]
        diversity = len(set(all_words)) / len(all_words) if all_words else 0
        return round(diversity, 4)
    except Exception:
        return 0.0


# ── Model Building ─────────────────────────────────────────────────────────────

def build_topic_model(
    n_neighbors: int = 15,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    cluster_selection_method: str = "eom",
    top_n_words: int = 10,
) -> BERTopic:
    """Construct a BERTopic model with specified hyperparameters."""

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )

    # Representation: multiple strategies
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    # Vectorizer: allow phrases up to 2 words
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
        max_df=0.85,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        verbose=False,
        calculate_probabilities=True,
    )

    return topic_model


# ── Grid Search ────────────────────────────────────────────────────────────────

def run_grid_search(
    docs: list[str],
    embeddings: np.ndarray,
    journal: str,
) -> pd.DataFrame:
    """
    Hyperparameter search over UMAP n_neighbors and HDBSCAN min_cluster_size.
    Selects best model by coherence score.
    """
    grid = {
        "n_neighbors": [10, 15, 20, 30],
        "min_cluster_size": [30, 50, 75, 100],
        "min_samples": [10],
        "cluster_selection_method": ["eom"],
    }

    results = []
    combos = list(product(*grid.values()))
    logger.info(f"Running grid search: {len(combos)} combinations")

    for combo in tqdm(combos, desc="Grid search"):
        params = dict(zip(grid.keys(), combo))
        try:
            model = build_topic_model(**params)
            topics, probs = model.fit_transform(docs, embeddings)

            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            noise_pct = (sum(t == -1 for t in topics) / len(topics)) * 100
            coherence = compute_coherence(model, docs)
            diversity = compute_diversity(model)

            result = {
                **params,
                "n_topics": n_topics,
                "noise_pct": round(noise_pct, 1),
                "coherence_npmi": coherence["coherence_npmi"],
                "diversity": diversity,
            }
            results.append(result)
            logger.info(f"  {params} → {n_topics} topics, NPMI={coherence['coherence_npmi']}, noise={noise_pct:.1f}%")
        except Exception as e:
            logger.warning(f"  {params} failed: {e}")
            continue

    df_results = pd.DataFrame(results)
    out_path = TABLE_DIR / f"grid_search_{journal}.csv"
    df_results.to_csv(out_path, index=False)
    logger.success(f"Grid search saved: {out_path}")
    return df_results


def select_best_params(df_grid: pd.DataFrame) -> dict:
    """
    Select best hyperparameters from grid search.
    Criterion: maximize coherence, with constraint noise_pct < 25%.
    """
    valid = df_grid[df_grid["noise_pct"] < 25.0].copy()
    if valid.empty:
        valid = df_grid.copy()
        logger.warning("All configs exceed 25% noise — relaxing constraint")

    best_row = valid.loc[valid["coherence_npmi"].idxmax()]
    best_params = {
        "n_neighbors": int(best_row["n_neighbors"]),
        "min_cluster_size": int(best_row["min_cluster_size"]),
        "min_samples": int(best_row["min_samples"]),
        "cluster_selection_method": best_row["cluster_selection_method"],
    }
    logger.info(f"Best params: {best_params}")
    logger.info(f"  n_topics={best_row['n_topics']}, NPMI={best_row['coherence_npmi']}, noise={best_row['noise_pct']}%")
    return best_params


# ── Main Run ───────────────────────────────────────────────────────────────────

def run_bertopic(args):
    # Load data
    df = pd.read_csv(args.corpus)
    embeddings = np.load(args.embeddings)
    docs = df["document"].tolist()
    journal = args.journal

    logger.info(f"Corpus: {len(docs)} documents | Embeddings: {embeddings.shape}")

    # Grid search or use defaults
    if args.grid_search:
        df_grid = run_grid_search(docs, embeddings, journal)
        best_params = select_best_params(df_grid)
    else:
        # Defaults — proven to work well for ~15-30k medical abstracts
        best_params = {
            "n_neighbors": 15,
            "min_cluster_size": 50,
            "min_samples": 10,
            "cluster_selection_method": "eom",
        }

    # Build and fit final model
    logger.info(f"Fitting final BERTopic model with params: {best_params}")
    topic_model = build_topic_model(**best_params)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Assign to dataframe
    df["topic_id"] = topics
    df["topic_probability"] = probs.max(axis=1) if probs.ndim > 1 else probs

    # Topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    noise_pct = (sum(t == -1 for t in topics) / len(topics)) * 100
    logger.info(f"Topics discovered: {n_topics}")
    logger.info(f"Noise documents: {noise_pct:.1f}%")

    # Evaluate
    coherence = compute_coherence(topic_model, docs)
    diversity = compute_diversity(topic_model)
    logger.info(f"Coherence (NPMI): {coherence['coherence_npmi']}")
    logger.info(f"Diversity: {diversity}")

    # Save model
    model_path = MODEL_DIR / f"bertopic_model_{journal}"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    logger.success(f"Model saved: {model_path}")

    # Save topic info table
    topic_info["coherence_npmi"] = coherence["coherence_npmi"]
    topic_info["diversity"] = diversity
    topic_info_path = TABLE_DIR / f"topic_info_{journal}.csv"
    topic_info.to_csv(topic_info_path, index=False)

    # Save per-document assignments
    doc_topics = df[["doc_id", "pmid", "year", "journal", "title", "topic_id", "topic_probability"]]
    doc_topics_path = TABLE_DIR / f"doc_topics_{journal}.csv"
    doc_topics.to_csv(doc_topics_path, index=False)

    # Save model config
    config = {
        **best_params,
        "journal": journal,
        "n_documents": len(docs),
        "n_topics": n_topics,
        "noise_pct": round(noise_pct, 2),
        "coherence_npmi": coherence["coherence_npmi"],
        "diversity": diversity,
    }
    config_path = MODEL_DIR / f"model_config_{journal}.json"
    config_path.write_text(json.dumps(config, indent=2))

    logger.success("Phase 6 complete.")
    return topic_model, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERTopic pipeline")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--journal", default="AJSM")
    parser.add_argument("--grid_search", action="store_true",
                        help="Run hyperparameter grid search before final fit")
    args = parser.parse_args()
    run_bertopic(args)
```

### 8.2 Execution

```bash
# Recommended: with grid search
python src/03_modeling/run_bertopic.py \
    --corpus data/processed/corpus_clean_AJSM.csv \
    --embeddings data/embeddings/embeddings_pubmedbert_AJSM.npy \
    --journal AJSM \
    --grid_search

# Quick run with defaults (for testing)
python src/03_modeling/run_bertopic.py \
    --corpus data/processed/corpus_clean_AJSM.csv \
    --embeddings data/embeddings/embeddings_pubmedbert_AJSM.npy \
    --journal AJSM
```

### 8.3 Optional LLM Topic Label Fine-tuning

After running the base model, optionally refine topic labels using GPT-4o:

**Create `src/03_modeling/label_topics_llm.py`:**

```python
"""
Optional: Use OpenAI GPT-4o to generate clinical topic labels.
Requires OPENAI_API_KEY in .env

Usage:
    python src/03_modeling/label_topics_llm.py --journal AJSM
"""

import os
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from bertopic import BERTopic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROMPT_TEMPLATE = """You are an expert sports medicine orthopedic surgeon reviewing research topics.

The following words describe a cluster of medical research abstracts:
{words}

Generate a concise clinical label (4-6 words maximum) that accurately describes 
the primary research theme. Respond with ONLY the label, no explanation.

Examples of good labels:
- "ACL reconstruction graft outcomes"
- "Rotator cuff repair techniques"  
- "Return to sport after knee injury"
- "Hip arthroscopy FAI treatment"
"""

def label_topics(journal: str = "AJSM"):
    model_path = f"outputs/models/bertopic_model_{journal}"
    topic_info_path = f"outputs/tables/topic_info_{journal}.csv"

    topic_model = BERTopic.load(model_path)
    topic_info = pd.read_csv(topic_info_path)
    client = OpenAI()

    labels = {}
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            labels[-1] = "Noise / Uncategorized"
            continue

        topic_id = row["Topic"]
        top_words = topic_model.get_topic(topic_id)
        words_str = ", ".join([w for w, _ in top_words[:15]])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(words=words_str)}],
            max_tokens=30,
            temperature=0.2,
        )
        label = response.choices[0].message.content.strip()
        labels[topic_id] = label
        logger.info(f"Topic {topic_id}: {label}")

    # Save labels
    topic_info["llm_label"] = topic_info["Topic"].map(labels)
    topic_info.to_csv(topic_info_path, index=False)
    labels_path = f"outputs/tables/topic_labels_llm_{journal}.json"
    Path(labels_path).write_text(json.dumps(labels, indent=2))
    logger.success(f"LLM labels saved to {labels_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="AJSM")
    args = parser.parse_args()
    label_topics(args.journal)
```

---

## 9. Phase 7 — Temporal Analysis

### 9.1 Overview
Classify topics as hot, cold, emerging, or stable based on publication frequency trends over time.

**Multiple-testing correction:** Uncorrected linear regressions across many topics inflate false-positive rates. Benjamini-Hochberg FDR correction is applied to all topic p-values before trend classification. All reported p-values and trend classifications use the adjusted `p_adj` column.

**Emerging definition:** Topic classified as Emerging requires (a) BH-adjusted p < 0.05, (b) positive slope, and (c) ≥60% of the topic's documents fall in the most recent 10 years. The 60% threshold is used consistently in both the code and manuscript.

**Create `src/04_analysis/temporal_analysis.py`:**

```python
"""
Phase 7: Temporal Trend Analysis
==================================
For each topic, fits a linear regression of document frequency vs. year
and classifies the trend. Also generates topics-over-time data for visualization.

Usage:
    python src/04_analysis/temporal_analysis.py --journal AJSM

Outputs:
    outputs/tables/temporal_trends_AJSM.csv
    outputs/tables/topics_over_time_AJSM.csv
    outputs/figures/temporal_heatmap_AJSM.html
"""

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from loguru import logger
from bertopic import BERTopic

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_FIGURES = Path("outputs/figures")
for d in [OUTPUT_TABLES, OUTPUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ── Trend Classification ───────────────────────────────────────────────────────

def classify_trend(slope: float, p_value: float, years: np.ndarray, counts: np.ndarray) -> str:
    """
    Classify a topic's temporal trend.

    Rules:
    - Emerging: significant positive slope AND >80% of documents in last 10 years
    - Hot: significant positive slope (p < 0.05)
    - Cold: significant negative slope (p < 0.05)
    - Stable: no significant trend
    """
    alpha = 0.05
    recent_years = years[years >= years.max() - 9]
    recent_counts = counts[years >= years.max() - 9]
    recent_pct = recent_counts.sum() / counts.sum() if counts.sum() > 0 else 0

    if p_value < alpha and slope > 0 and recent_pct >= 0.6:
        return "Emerging"
    elif p_value < alpha and slope > 0:
        return "Hot"
    elif p_value < alpha and slope < 0:
        return "Cold"
    else:
        return "Stable"


def compute_topic_trends(
    df: pd.DataFrame,
    min_year: int = 1995,
    max_year: int = 2025,
) -> pd.DataFrame:
    """
    Compute linear trend statistics for each topic.
    Returns a DataFrame with one row per topic.
    """
    year_range = np.arange(min_year, max_year + 1)
    results = []

    topic_ids = sorted(df["topic_id"].unique())
    topic_ids = [t for t in topic_ids if t != -1]

    for topic_id in topic_ids:
        topic_docs = df[df["topic_id"] == topic_id]

        # Count documents per year
        year_counts = topic_docs.groupby("year").size().reindex(year_range, fill_value=0)
        counts = year_counts.values.astype(float)

        # Normalize to proportion of total publications per year (controls for journal growth)
        total_per_year = df.groupby("year").size().reindex(year_range, fill_value=1)
        proportions = counts / total_per_year.values * 100  # as percentage

        # Linear regression on proportions
        slope, intercept, r_value, p_value, std_err = stats.linregress(year_range, proportions)

        # 5-year moving average for smoothed trend
        smoothed = pd.Series(proportions).rolling(window=5, center=True, min_periods=1).mean().values

        trend_class = classify_trend(slope, p_value, year_range, counts)

        results.append({
            "topic_id": topic_id,
            "n_documents": len(topic_docs),
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_value**2, 4),
            "p_value": round(p_value, 6),
            "trend_class": trend_class,
            "mean_pct": round(proportions.mean(), 4),
            "peak_year": int(year_range[np.argmax(counts)]),
            "peak_count": int(counts.max()),
            "recent_5yr_pct": round(proportions[-5:].mean(), 4),
            "early_5yr_pct": round(proportions[:5].mean(), 4),
        })

    df_trends = pd.DataFrame(results)
    return df_trends


def run_temporal_analysis(journal: str = "AJSM"):
    # Load data
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")

    logger.info(f"Running temporal analysis for {journal}: {len(doc_topics)} documents")

    # Compute trends
    df_trends = compute_topic_trends(doc_topics)

    # Merge with topic labels
    topic_labels = topic_info[["Topic", "Name", "Count"]].rename(
        columns={"Topic": "topic_id", "Name": "topic_name", "Count": "topic_count"}
    )
    df_trends = df_trends.merge(topic_labels, on="topic_id", how="left")

    # Add LLM labels if available
    llm_path = OUTPUT_TABLES / f"topic_labels_llm_{journal}.json"
    if llm_path.exists():
        import json
        llm_labels = json.loads(llm_path.read_text())
        df_trends["llm_label"] = df_trends["topic_id"].map(
            {int(k): v for k, v in llm_labels.items()}
        )

    # Sort by trend class then slope
    trend_order = {"Emerging": 0, "Hot": 1, "Stable": 2, "Cold": 3}
    df_trends["trend_order"] = df_trends["trend_class"].map(trend_order)
    df_trends = df_trends.sort_values(["trend_order", "slope"], ascending=[True, False])

    # Save
    out_path = OUTPUT_TABLES / f"temporal_trends_{journal}.csv"
    df_trends.to_csv(out_path, index=False)
    logger.success(f"Temporal trends saved: {out_path}")

    # Topics over time (using BERTopic built-in)
    docs = doc_topics["document"].tolist() if "document" in doc_topics.columns else None
    timestamps = doc_topics["year"].tolist()

    topics_over_time = model.topics_over_time(
        doc_topics.get("document", doc_topics["title"]).tolist(),
        timestamps,
        nr_bins=10,
        evolution_tuning=True,
        global_tuning=True,
    )
    tot_path = OUTPUT_TABLES / f"topics_over_time_{journal}.csv"
    topics_over_time.to_csv(tot_path, index=False)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL TREND SUMMARY")
    logger.info("="*60)
    for trend_class in ["Emerging", "Hot", "Stable", "Cold"]:
        subset = df_trends[df_trends["trend_class"] == trend_class]
        logger.info(f"\n{trend_class} Topics ({len(subset)}):")
        for _, row in subset.iterrows():
            label = row.get("llm_label") or row.get("topic_name", f"Topic {row['topic_id']}")
            logger.info(f"  [{row['topic_id']}] {label} | n={row['n_documents']} | slope={row['slope']:.4f} | p={row['p_value']:.4f}")

    return df_trends, topics_over_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="AJSM")
    args = parser.parse_args()
    run_temporal_analysis(args.journal)
```

### 9.2 Execution

```bash
python src/04_analysis/temporal_analysis.py --journal AJSM
```

---

## 10. Phase 8 — Validation

### 10.1 Validation Script

**Create `src/04_analysis/validate.py`:**

```python
"""
Phase 8: Topic Model Validation
=================================
Computes all validation metrics and generates the validation report.

Metrics computed:
1. Topic coherence (NPMI) — internal semantic quality
2. Topic diversity — uniqueness of words across topics
3. Silhouette score — cluster separation quality
4. Noise ratio — proportion of uncategorized documents
5. Inter-rater reliability setup — exports topics for manual review

Usage:
    python src/04_analysis/validate.py --journal AJSM

Outputs:
    outputs/tables/validation_report_AJSM.csv
    outputs/tables/manual_review_sheet_AJSM.xlsx  (for human expert review)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from loguru import logger
from bertopic import BERTopic

OUTPUT_TABLES = Path("outputs/tables")


def compute_silhouette(embeddings: np.ndarray, topics: list[int], sample_size: int = 5000) -> float:
    """Compute silhouette score on a subsample (full computation is O(n^2))."""
    try:
        topic_arr = np.array(topics)
        # Exclude noise
        valid_idx = np.where(topic_arr != -1)[0]
        if len(valid_idx) < 100:
            return None

        # Subsample for efficiency
        if len(valid_idx) > sample_size:
            sample_idx = np.random.choice(valid_idx, size=sample_size, replace=False)
        else:
            sample_idx = valid_idx

        score = silhouette_score(
            embeddings[sample_idx],
            topic_arr[sample_idx],
            metric="cosine",
            sample_size=min(sample_size, len(sample_idx)),
        )
        return round(float(score), 4)
    except Exception as e:
        logger.warning(f"Silhouette computation failed: {e}")
        return None


def generate_review_sheet(topic_model: BERTopic, topic_info: pd.DataFrame, journal: str):
    """
    Generate an Excel sheet for expert manual review of topic labels.
    Two reviewers independently label each topic; agreement computed later.
    """
    topics = topic_info[topic_info["Topic"] != -1].copy()

    review_rows = []
    for _, row in topics.iterrows():
        topic_id = row["Topic"]
        top_words = topic_model.get_topic(topic_id)
        words_str = ", ".join([w for w, _ in top_words[:15]])
        review_rows.append({
            "topic_id": topic_id,
            "n_documents": row["Count"],
            "top_15_words": words_str,
            "bertopic_name": row["Name"],
            "reviewer_1_label": "",  # To be filled by expert
            "reviewer_2_label": "",  # To be filled by expert
            "notes": "",
        })

    df_review = pd.DataFrame(review_rows)
    out_path = OUTPUT_TABLES / f"manual_review_sheet_{journal}.xlsx"
    df_review.to_excel(out_path, index=False, engine="openpyxl")
    logger.info(f"Expert review sheet: {out_path}")
    return df_review


def run_validation(journal: str = "AJSM"):
    model_path = f"outputs/models/bertopic_model_{journal}"
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")

    import glob
    emb_files = sorted(glob.glob(f"data/embeddings/embeddings_*_{journal}.npy"))
    embeddings = np.load(emb_files[0]) if emb_files else None

    model = BERTopic.load(model_path)
    topics = doc_topics["topic_id"].tolist()

    # 1. Coherence
    from run_bertopic import compute_coherence, compute_diversity
    docs = doc_topics.get("document", doc_topics["title"]).tolist()
    coherence = compute_coherence(model, docs)
    diversity = compute_diversity(model)

    # 2. Noise ratio
    noise_ratio = round(sum(t == -1 for t in topics) / len(topics) * 100, 2)

    # 3. Topic count
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)

    # 4. Silhouette (if embeddings available)
    silhouette = compute_silhouette(embeddings, topics) if embeddings is not None else None

    # 5. Summary stats
    topic_sizes = topic_info[topic_info["Topic"] != -1]["Count"]

    report = {
        "journal": journal,
        "n_documents": len(docs),
        "n_topics": n_topics,
        "noise_pct": noise_ratio,
        "coherence_npmi": coherence["coherence_npmi"],
        "diversity": diversity,
        "silhouette_score": silhouette,
        "median_topic_size": int(topic_sizes.median()),
        "min_topic_size": int(topic_sizes.min()),
        "max_topic_size": int(topic_sizes.max()),
    }

    logger.info("\nVALIDATION REPORT")
    logger.info("="*50)
    for k, v in report.items():
        logger.info(f"  {k:30s}: {v}")

    # Save
    pd.DataFrame([report]).to_csv(OUTPUT_TABLES / f"validation_report_{journal}.csv", index=False)

    # Generate review sheet
    generate_review_sheet(model, topic_info, journal)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="AJSM")
    args = parser.parse_args()
    run_validation(args.journal)
```

---

## 11. Phase 9 — Visualization

### 11.1 Visualization Script

**Create `src/05_visualization/generate_figures.py`:**

```python
"""
Phase 9: Visualization
=======================
Generates all publication-ready figures.

Figures produced:
  Fig 1: Intertopic distance map (2D UMAP)
  Fig 2: Top words per topic (bar chart)
  Fig 3: Topics over time (line chart)
  Fig 4: Temporal trend heatmap (topics x years)
  Fig 5: Topic size distribution
  Fig 6: Hot/Cold/Emerging/Stable classification summary
  Fig 7: Hierarchical topic tree (dendrogram)

Usage:
    python src/05_visualization/generate_figures.py --journal AJSM

Outputs:
    outputs/figures/fig1_topic_map_AJSM.html
    outputs/figures/fig2_top_words_AJSM.html
    outputs/figures/fig3_topics_over_time_AJSM.html
    outputs/figures/fig4_temporal_heatmap_AJSM.html
    outputs/figures/fig5_topic_sizes_AJSM.html
    outputs/figures/fig6_trend_summary_AJSM.html
    outputs/figures/fig7_hierarchy_AJSM.html
"""

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from bertopic import BERTopic
from loguru import logger

OUTPUT_FIGURES = Path("outputs/figures")
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES = Path("outputs/tables")

TREND_COLORS = {
    "Emerging": "#2ECC71",
    "Hot": "#E74C3C",
    "Stable": "#95A5A6",
    "Cold": "#3498DB",
}


def load_data(journal: str):
    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    temporal_trends = pd.read_csv(OUTPUT_TABLES / f"temporal_trends_{journal}.csv")

    tot_path = OUTPUT_TABLES / f"topics_over_time_{journal}.csv"
    topics_over_time = pd.read_csv(tot_path) if tot_path.exists() else None

    return model, topic_info, doc_topics, temporal_trends, topics_over_time


def fig1_topic_map(model, journal):
    """Intertopic distance map using BERTopic built-in."""
    fig = model.visualize_topics(title=f"<b>Topic Map — {journal}</b>")
    fig.write_html(OUTPUT_FIGURES / f"fig1_topic_map_{journal}.html")
    logger.info("Fig 1: Topic map saved")


def fig2_top_words(model, topic_info, journal, top_n=20):
    """Bar chart of top words for top N topics by document count."""
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")["Topic"].tolist()
    fig = model.visualize_barchart(
        topics=top_topics,
        top_n_topics=top_n,
        n_words=10,
        title=f"<b>Top Words per Topic — {journal}</b>",
    )
    fig.write_html(OUTPUT_FIGURES / f"fig2_top_words_{journal}.html")
    logger.info("Fig 2: Top words saved")


def fig3_topics_over_time(model, topics_over_time, temporal_trends, journal, top_n=15):
    """Topics over time line chart for top N topics."""
    if topics_over_time is None:
        logger.warning("Topics over time data not available — skipping Fig 3")
        return

    top_topics = temporal_trends.nlargest(top_n, "n_documents")["topic_id"].tolist()
    fig = model.visualize_topics_over_time(
        topics_over_time,
        top_n_topics=top_n,
        topics=top_topics,
        title=f"<b>Topics Over Time — {journal}</b>",
    )
    fig.write_html(OUTPUT_FIGURES / f"fig3_topics_over_time_{journal}.html")
    logger.info("Fig 3: Topics over time saved")


def fig4_temporal_heatmap(doc_topics, temporal_trends, journal):
    """Heatmap: topics (rows) x year bins (cols), color = normalized frequency."""
    # Create year bins (5-year bins)
    doc_topics = doc_topics.copy()
    doc_topics["year_bin"] = (doc_topics["year"] // 5 * 5).astype(str) + "–" + \
                              ((doc_topics["year"] // 5 * 5) + 4).astype(str)

    # Pivot table: topic x year_bin
    pivot = doc_topics[doc_topics["topic_id"] != -1].groupby(
        ["topic_id", "year_bin"]
    ).size().unstack(fill_value=0)

    # Normalize by row (within-topic)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Get topic labels
    label_map = {}
    if "llm_label" in temporal_trends.columns:
        label_map = temporal_trends.set_index("topic_id")["llm_label"].dropna().to_dict()
    elif "topic_name" in temporal_trends.columns:
        label_map = temporal_trends.set_index("topic_id")["topic_name"].to_dict()

    pivot_norm.index = [label_map.get(i, f"Topic {i}") for i in pivot_norm.index]

    # Sort by trend (Emerging → Hot → Stable → Cold)
    trend_map = temporal_trends.set_index("topic_id")["trend_class"].to_dict()
    trend_order = {"Emerging": 0, "Hot": 1, "Stable": 2, "Cold": 3}

    sorted_idx = sorted(
        pivot_norm.index,
        key=lambda x: trend_order.get(
            trend_map.get(
                next((k for k, v in label_map.items() if v == x), -1), "Stable"
            ), 2
        )
    )
    pivot_norm = pivot_norm.loc[sorted_idx]

    fig = px.imshow(
        pivot_norm,
        labels=dict(x="Year Range", y="Topic", color="% of Topic Docs"),
        title=f"<b>Research Activity Heatmap — {journal}</b>",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=max(400, len(pivot_norm) * 20 + 100))
    fig.write_html(OUTPUT_FIGURES / f"fig4_temporal_heatmap_{journal}.html")
    logger.info("Fig 4: Temporal heatmap saved")


def fig5_topic_sizes(temporal_trends, journal):
    """Bar chart of topic sizes, colored by trend class."""
    df = temporal_trends.sort_values("n_documents", ascending=False).head(30).copy()
    label_col = "llm_label" if "llm_label" in df.columns else "topic_name"
    df["label"] = df[label_col].fillna(df["topic_id"].astype(str))

    fig = px.bar(
        df,
        x="label",
        y="n_documents",
        color="trend_class",
        color_discrete_map=TREND_COLORS,
        title=f"<b>Topic Size by Trend Class — {journal}</b>",
        labels={"label": "Topic", "n_documents": "Number of Documents"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    fig.write_html(OUTPUT_FIGURES / f"fig5_topic_sizes_{journal}.html")
    logger.info("Fig 5: Topic sizes saved")


def fig6_trend_summary(temporal_trends, journal):
    """Summary figure: scatter of slope vs. document count, colored by trend."""
    df = temporal_trends.copy()
    label_col = "llm_label" if "llm_label" in df.columns else "topic_name"
    df["label"] = df[label_col].fillna(df["topic_id"].astype(str))

    fig = px.scatter(
        df,
        x="slope",
        y="n_documents",
        color="trend_class",
        color_discrete_map=TREND_COLORS,
        text="label",
        size="n_documents",
        title=f"<b>Topic Trends: Growth Rate vs. Volume — {journal}</b>",
        labels={"slope": "Annual Growth Rate (% publications)", "n_documents": "Total Documents"},
        hover_data=["p_value", "r_squared", "peak_year"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=700)
    fig.write_html(OUTPUT_FIGURES / f"fig6_trend_summary_{journal}.html")
    logger.info("Fig 6: Trend summary saved")


def fig7_hierarchy(model, journal):
    """Hierarchical topic clustering dendrogram."""
    fig = model.visualize_hierarchy(title=f"<b>Topic Hierarchy — {journal}</b>")
    fig.write_html(OUTPUT_FIGURES / f"fig7_hierarchy_{journal}.html")
    logger.info("Fig 7: Hierarchy saved")


def generate_all_figures(journal: str = "AJSM"):
    logger.info(f"Generating all figures for {journal}...")
    model, topic_info, doc_topics, temporal_trends, topics_over_time = load_data(journal)

    fig1_topic_map(model, journal)
    fig2_top_words(model, topic_info, journal)
    fig3_topics_over_time(model, topics_over_time, temporal_trends, journal)
    fig4_temporal_heatmap(doc_topics, temporal_trends, journal)
    fig5_topic_sizes(temporal_trends, journal)
    fig6_trend_summary(temporal_trends, journal)
    fig7_hierarchy(model, journal)

    logger.success(f"All figures saved to outputs/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="AJSM")
    args = parser.parse_args()
    generate_all_figures(args.journal)
```

---

## 12. Phase 10 — Reporting

### 12.1 Summary Tables for Manuscript

**Create `src/05_visualization/generate_tables.py`:**

```python
"""
Phase 10: Generate publication-ready tables for manuscript.

Tables produced:
  Table 1: Corpus characteristics (N per year, total)
  Table 2: All topics with labels, size, trend class, top words
  Table 3: Hot topics (top 10 by slope, p<0.05)
  Table 4: Cold topics (top 10 by slope magnitude, p<0.05)
  Table 5: Emerging topics
  Table 6: Validation metrics comparison (BERTopic vs. LDA baseline)

Usage:
    python src/05_visualization/generate_tables.py --journal AJSM
"""

import argparse
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from loguru import logger

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_REPORTS = Path("outputs/reports")
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def generate_all_tables(journal: str = "AJSM"):
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    temporal_trends = pd.read_csv(OUTPUT_TABLES / f"temporal_trends_{journal}.csv")
    validation = pd.read_csv(OUTPUT_TABLES / f"validation_report_{journal}.csv")
    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")

    label_col = "llm_label" if "llm_label" in temporal_trends.columns else "topic_name"

    # ── Table 1: Corpus Summary ────────────────────────────────────────────────
    t1 = doc_topics.groupby("year").size().reset_index(name="n_publications")
    t1["cumulative"] = t1["n_publications"].cumsum()
    t1.to_csv(OUTPUT_REPORTS / f"table1_corpus_{journal}.csv", index=False)

    # ── Table 2: All Topics ────────────────────────────────────────────────────
    topics_all = temporal_trends.merge(
        topic_info[["Topic", "Name"]].rename(columns={"Topic": "topic_id"}),
        on="topic_id", how="left"
    )
    top_words_map = {
        t_id: ", ".join([w for w, _ in model.get_topic(t_id)[:10]])
        for t_id in temporal_trends["topic_id"]
    }
    topics_all["top_10_words"] = topics_all["topic_id"].map(top_words_map)
    topics_all = topics_all[[
        "topic_id", label_col, "n_documents", "trend_class",
        "slope", "p_value", "r_squared", "peak_year", "top_10_words"
    ]].rename(columns={label_col: "topic_label"})
    topics_all.to_csv(OUTPUT_REPORTS / f"table2_all_topics_{journal}.csv", index=False)

    # ── Table 3: Hot Topics ────────────────────────────────────────────────────
    hot = temporal_trends[temporal_trends["trend_class"].isin(["Hot", "Emerging"])].copy()
    hot = hot.sort_values("slope", ascending=False).head(15)
    hot["top_words"] = hot["topic_id"].map(top_words_map)
    hot[[label_col, "n_documents", "slope", "p_value", "peak_year", "recent_5yr_pct", "top_words"]].to_csv(
        OUTPUT_REPORTS / f"table3_hot_topics_{journal}.csv", index=False
    )

    # ── Table 4: Cold Topics ───────────────────────────────────────────────────
    cold = temporal_trends[temporal_trends["trend_class"] == "Cold"].copy()
    cold = cold.sort_values("slope", ascending=True).head(15)
    cold["top_words"] = cold["topic_id"].map(top_words_map)
    cold[[label_col, "n_documents", "slope", "p_value", "peak_year", "early_5yr_pct", "top_words"]].to_csv(
        OUTPUT_REPORTS / f"table4_cold_topics_{journal}.csv", index=False
    )

    # ── Table 5: Validation ────────────────────────────────────────────────────
    validation.to_csv(OUTPUT_REPORTS / f"table5_validation_{journal}.csv", index=False)

    logger.success(f"All tables saved to outputs/reports/")
    logger.info(f"\nSummary for {journal}:")
    logger.info(f"  Total documents: {len(doc_topics)}")
    logger.info(f"  Topics discovered: {temporal_trends['topic_id'].nunique()}")
    logger.info(f"  Hot topics: {len(temporal_trends[temporal_trends['trend_class']=='Hot'])}")
    logger.info(f"  Emerging topics: {len(temporal_trends[temporal_trends['trend_class']=='Emerging'])}")
    logger.info(f"  Cold topics: {len(temporal_trends[temporal_trends['trend_class']=='Cold'])}")
    logger.info(f"  Stable topics: {len(temporal_trends[temporal_trends['trend_class']=='Stable'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="AJSM")
    args = parser.parse_args()
    generate_all_tables(args.journal)
```

---

## 11b. Phase 6b — VOSviewer Export (Comparison Method)

Run this after preprocessing (Phase 2) is complete to export the corpus for traditional bibliometric comparison.

```bash
# Export author keywords and RIS file for Arthroscopy
python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY

# Alternative: use MeSH terms instead of author keywords
python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY --keyword_type mesh
```

Outputs:
- `outputs/vosviewer/vosviewer_ris_ARTHROSCOPY.ris` — import directly into VOSviewer
- `outputs/vosviewer/keyword_cooccurrence_ARTHROSCOPY.csv` — pre-computed co-occurrence matrix
- `outputs/vosviewer/vosviewer_readme_ARTHROSCOPY.txt` — step-by-step import instructions

**Comparison methodology:** For each VOSviewer cluster, compute Jaccard similarity against the top-10 keywords of each BERTopic topic. A topic is "matched" at J≥0.20. Report match rate, mean Jaccard, and a qualitative visual overlay. See `export_for_vosviewer.py` docstring for manuscript language template.

---

## 13. Configuration Reference

### 13.1 Master Config File

The `files/study_config.yaml` file documents all hyperparameters as a reference. Scripts use hard-coded defaults that match these values exactly. To change a hyperparameter, edit both the config file (for documentation) and the corresponding script argument (for execution).

```yaml
# files/study_config.yaml — reference only; scripts read CLI arguments, not this file
study:
  title: "BERTopic Analysis of Sports Medicine Orthopedic Surgery Literature"
  version: "1.0.0"
  journal: "ARTHROSCOPY"

data:
  start_year: 1995
  end_year: 2025
  min_abstract_words: 50
  min_abstract_chars: 200
  target_journals:
    AJSM:
      pubmed_query: '"Am J Sports Med"[Journal]'
      full_name: "American Journal of Sports Medicine"
    ARTHROSCOPY:
      pubmed_query: '"Arthroscopy"[Journal]'
      full_name: "Arthroscopy: Journal of Arthroscopic and Related Surgery"
    KSSTA:
      pubmed_query: '"Knee Surg Sports Traumatol Arthrosc"[Journal]'
      full_name: "Knee Surgery, Sports Traumatology, Arthroscopy"

embedding:
  primary_model: "mpnet"
  comparison_model: "pubmedbert"
  batch_size: 32
  normalize: true

umap:
  n_neighbors: 15
  n_components: 5
  min_dist: 0.0
  metric: "cosine"
  random_state: 42

hdbscan:
  min_cluster_size: 50
  min_samples: 10
  metric: "euclidean"
  cluster_selection_method: "eom"

grid_search:
  enabled: true
  n_neighbors: [10, 15, 20, 30]
  min_cluster_size: [30, 50, 75, 100]
  min_samples: [10]
  max_noise_pct: 25.0

representation:
  top_n_words: 10
  ngram_range: [1, 2]
  use_llm_labels: false   # Set true if OPENAI_API_KEY is set

temporal:
  trend_alpha: 0.05       # Significance threshold
  emerging_recent_pct: 0.6  # Minimum recent proportion for "Emerging"

validation:
  coherence_metric: "c_npmi"
  silhouette_sample_size: 5000

output:
  save_model: true
  generate_figures: true
  generate_tables: true
  figure_format: "html"
```

### 13.2 Run Order (Full Pipeline)

```bash
# 0. Setup
python src/00_verify_setup.py

# 1. Data Collection
python src/01_data_collection/fetch_pubmed.py --journal ARTHROSCOPY

# 2. Preprocessing
python src/02_preprocessing/preprocess.py \
    --input "data/raw/pubmed_raw_ARTHROSCOPY_*.csv" \
    --journal ARTHROSCOPY

# 2b. VOSviewer export (run after preprocessing, before embedding — no GPU needed)
python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY

# 3. Embedding (primary model — reports truncation rate automatically)
python src/03_modeling/embed.py \
    --input data/processed/corpus_clean_ARTHROSCOPY.csv \
    --model mpnet

# 4-6. BERTopic (with grid search)
python src/03_modeling/run_bertopic.py \
    --corpus data/processed/corpus_clean_ARTHROSCOPY.csv \
    --embeddings data/embeddings/embeddings_mpnet_ARTHROSCOPY.npy \
    --journal ARTHROSCOPY \
    --grid_search

# Optional: LLM topic labeling (requires: pip install openai==1.40.0 and OPENAI_API_KEY in .env)
# python src/03_modeling/label_topics_llm.py --journal ARTHROSCOPY

# 7. Temporal analysis (includes BH FDR correction)
python src/04_analysis/temporal_analysis.py --journal ARTHROSCOPY

# 8. Validation
python src/04_analysis/validate.py --journal ARTHROSCOPY

# 9. Figures
python src/05_visualization/generate_figures.py --journal ARTHROSCOPY

# 10. Tables
python src/05_visualization/generate_tables.py --journal ARTHROSCOPY
```

---

## 14. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|---------|
| `CUDA out of memory` | Batch size too large | Reduce `--batch_size` to 16 or 8 |
| Too many topics (>100) | `min_cluster_size` too small | Increase to 75 or 100 |
| Too few topics (<5) | `min_cluster_size` too large | Decrease to 30 or 20 |
| >40% noise docs | `min_samples` too high | Reduce `min_samples` to 5 |
| PubMed rate limit error | Too many requests | Ensure `PUBMED_API_KEY` is set; add `time.sleep(1)` |
| `KeyError: 'document'` in validate.py | Column not in doc_topics | Merge corpus back in before validating |
| Low coherence (<0.1) | Poor embeddings or too many topics | Switch embedding model (e.g. `pubmedbert`) or increase `min_cluster_size` |
| UMAP takes >30 min | Too many docs without GPU | Set `low_memory=True` in UMAP config |

### Reproducibility Checklist
- [ ] `random_state=42` set in UMAP
- [ ] `requirements.txt` pinned to exact versions
- [ ] Raw PubMed query and retrieval date logged
- [ ] Grid search results saved before final model fit
- [ ] Model saved with `serialization="safetensors"`

---

## 15. Expected Outputs Checklist

```
outputs/
├── models/
│   ├── bertopic_model_AJSM/          ← Saved BERTopic model (loadable)
│   └── model_config_AJSM.json        ← Hyperparameters used
├── tables/
│   ├── topic_info_AJSM.csv           ← All topics: ID, name, count, words
│   ├── doc_topics_AJSM.csv           ← Per-document: topic ID, probability
│   ├── temporal_trends_AJSM.csv      ← Per-topic: slope, p, trend class
│   ├── topics_over_time_AJSM.csv     ← BERTopic time series data
│   ├── validation_report_AJSM.csv    ← All validation metrics
│   ├── grid_search_AJSM.csv          ← Hyperparameter search results
│   └── manual_review_sheet_AJSM.xlsx ← For expert review / IRR
├── figures/
│   ├── fig1_topic_map_AJSM.html
│   ├── fig2_top_words_AJSM.html
│   ├── fig3_topics_over_time_AJSM.html
│   ├── fig4_temporal_heatmap_AJSM.html
│   ├── fig5_topic_sizes_AJSM.html
│   ├── fig6_trend_summary_AJSM.html
│   └── fig7_hierarchy_AJSM.html
└── reports/
    ├── table1_corpus_AJSM.csv
    ├── table2_all_topics_AJSM.csv
    ├── table3_hot_topics_AJSM.csv
    ├── table4_cold_topics_AJSM.csv
    └── table5_validation_AJSM.csv
```

---

*Protocol version 1.0 — Created for BERTopic Sports Medicine Orthopedic Surgery Study*
*To extend to multi-journal analysis: repeat Phases 1–8 for each journal, then run a combined corpus analysis using `--journal ALL`.*
