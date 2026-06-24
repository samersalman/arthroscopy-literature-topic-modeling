"""
Phase 1: PubMed Abstract Retrieval
===================================
Fetches all qualifying abstracts from target journals via NCBI Entrez API.

Usage:
    python src/01_data_collection/fetch_pubmed.py --journal ARTHROSCOPY
    python src/01_data_collection/fetch_pubmed.py --journal ALL
    python src/01_data_collection/fetch_pubmed.py --journal ARTHROSCOPY --start_year 2000 --end_year 2025

Outputs:
    data/raw/pubmed_raw_ARTHROSCOPY_YYYYMMDD.csv
    data/raw/fetch_log_YYYYMMDD.txt
"""

import os
import sys
import time
import argparse
from typing import Optional
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
# Only set api_key if non-empty; an empty string adds "&api_key=" to the URL and NCBI returns HTTP 400
_api_key = os.getenv("PUBMED_API_KEY", "").strip()
if _api_key:
    Entrez.api_key = _api_key

# ISSN variants prevent missed records due to PubMed abbreviation differences
JOURNAL_QUERIES = {
    "AJSM":        '("Am J Sports Med"[Journal] OR "0363-5465"[ISSN])',
    "ARTHROSCOPY": '("Arthroscopy"[Journal] OR "0749-8063"[ISSN])',
    "KSSTA":       '("Knee Surg Sports Traumatol Arthrosc"[Journal] OR "0942-2056"[ISSN])',
    "JISAKOS":     '("J ISAKOS"[Journal] OR "2059-7754"[ISSN])',
    "OJSM":        '("Orthop J Sports Med"[Journal] OR "2325-9671"[ISSN])',
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add file log so fetch queries are permanently recorded for reproducibility
_log_path = OUTPUT_DIR / f"fetch_log_{datetime.now().strftime('%Y%m%d')}.txt"
logger.add(str(_log_path), format="{time} | {level} | {message}", level="INFO")

# ── Helper Functions ───────────────────────────────────────────────────────────

def build_query(journal_key: str, start_year: int, end_year: int) -> str:
    """Construct PubMed search query for a journal and date range."""
    journal_term = JOURNAL_QUERIES[journal_key]
    date_term = f'("{start_year}/01/01"[PDAT] : "{end_year}/12/31"[PDAT])'
    return f'{journal_term} AND {date_term} AND hasabstract[text] AND English[lang]'


def search_pubmed(query: str, retmax: int = 100000, max_retries: int = 6) -> list[str]:
    """Return list of PMIDs matching the query, retrying transient NCBI backend errors."""
    logger.info(f"Searching PubMed: {query}")
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, usehistory="y")
            record = Entrez.read(handle)
            handle.close()
            pmids = record["IdList"]
            logger.info(f"Found {len(pmids)} records")
            return pmids
        except Exception as e:
            last_exc = e
            wait = min(60, 5 * attempt)
            logger.warning(f"esearch attempt {attempt}/{max_retries} failed ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)
    logger.error(f"esearch failed after {max_retries} attempts: {last_exc}")
    raise last_exc


def fetch_records_batch(pmids: list[str], batch_size: int = 200, max_retries: int = 4) -> list[dict]:
    """Fetch full records for a list of PMIDs in batches, retrying transient errors per batch."""
    records = []
    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching batches"):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        for attempt in range(1, max_retries + 1):
            try:
                handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
                batch_records = Entrez.read(handle)
                handle.close()
                for article in batch_records["PubmedArticle"]:
                    parsed = parse_article(article)
                    if parsed:
                        records.append(parsed)
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Batch {i//batch_size} failed after {max_retries} attempts: {e}")
                else:
                    logger.warning(f"Batch {i//batch_size} attempt {attempt}/{max_retries} failed: {e}; retrying")
                time.sleep(min(60, 5 * attempt))
        # Rate limiting: 10 req/s with API key, 3 req/s without
        sleep_time = 0.11 if Entrez.api_key else 0.34
        time.sleep(sleep_time)
    return records


def parse_article(article: dict) -> Optional[dict]:
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
                abstract_text = " ".join([
                    f"{getattr(sec, 'attributes', {}).get('Label', '')}: {str(sec)}"
                    for sec in abstract_obj
                ]).strip()
            else:
                abstract_text = str(abstract_obj).strip()

        if not abstract_text:
            return None

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
    parser.add_argument("--journal", default="ARTHROSCOPY",
                        choices=list(JOURNAL_QUERIES.keys()) + ["ALL"],
                        help="Journal to fetch (default: ARTHROSCOPY)")
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
