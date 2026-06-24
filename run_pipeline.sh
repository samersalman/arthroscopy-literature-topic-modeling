#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BERTopic Sports Medicine Study — Full Pipeline
# Run from the project root with the venv active:
#   source venv/bin/activate
#   bash run_pipeline.sh [JOURNAL]
#
# Default journal: ARTHROSCOPY
# Override:        bash run_pipeline.sh AJSM
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

JOURNAL="${1:-ARTHROSCOPY}"
echo "=================================================="
echo " BERTopic Sports Medicine Study"
echo " Journal: $JOURNAL"
echo " Started: $(date)"
echo "=================================================="

# ── Guard: .env file must exist and PUBMED_EMAIL must be filled ───────────────
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: .env file not found."
    echo "  Copy .env.template to .env and fill in your PUBMED_EMAIL:"
    echo "    cp files/.env.template .env"
    echo ""
    exit 1
fi

if grep -q "your_email@institution.edu" .env; then
    echo ""
    echo "ERROR: PUBMED_EMAIL is still set to the placeholder in .env."
    echo "  Open .env and replace 'your_email@institution.edu' with your real address."
    echo ""
    exit 1
fi

if ! grep -q "PUBMED_EMAIL=." .env; then
    echo ""
    echo "ERROR: PUBMED_EMAIL is empty in .env. Add your email before running."
    echo ""
    exit 1
fi

# ── Phase 0: Verify setup ─────────────────────────────────────────────────────
echo ""
echo "── Phase 0: Verifying setup ──────────────────────────────────────────────"
python src/00_verify_setup.py

# ── Phase 1: Data collection ──────────────────────────────────────────────────
echo ""
echo "── Phase 1: Fetching abstracts from PubMed (this takes ~30–40 min) ───────"
python src/01_data_collection/fetch_pubmed.py --journal "$JOURNAL"
python src/01_data_collection/check_raw_data.py \
    --input "data/raw/pubmed_raw_${JOURNAL}_*.csv"

# ── Phase 2: Preprocessing ────────────────────────────────────────────────────
echo ""
echo "── Phase 2: Preprocessing corpus ────────────────────────────────────────"
python src/02_preprocessing/preprocess.py \
    --input "data/raw/pubmed_raw_${JOURNAL}_*.csv" \
    --journal "$JOURNAL"

# ── Phase 2b: VOSviewer export ────────────────────────────────────────────────
echo ""
echo "── Phase 2b: Exporting for VOSviewer comparison ─────────────────────────"
python src/06_vosviewer/export_for_vosviewer.py --journal "$JOURNAL"

# ── Phase 3: Embedding ────────────────────────────────────────────────────────
echo ""
echo "── Phase 3: Generating MPNet embeddings (slow on CPU) ─────────────"
python src/03_modeling/embed.py \
    --input "data/processed/corpus_clean_${JOURNAL}.csv" \
    --model mpnet \
    --batch_size 32 \
    --journal "$JOURNAL"

# ── Phases 4–6: BERTopic modeling ────────────────────────────────────────────
echo ""
echo "── Phases 4-6: BERTopic modeling with hyperparameter grid search ────────"
python src/03_modeling/run_bertopic.py \
    --corpus "data/processed/corpus_clean_${JOURNAL}.csv" \
    --embeddings "data/embeddings/embeddings_mpnet_${JOURNAL}.npy" \
    --journal "$JOURNAL" \
    --grid_search

# ── Phase 7: Temporal analysis ────────────────────────────────────────────────
echo ""
echo "── Phase 7: Temporal trend analysis (BH-FDR corrected) ─────────────────"
python src/04_analysis/temporal_analysis.py --journal "$JOURNAL"

# ── Phase 8: Validation ───────────────────────────────────────────────────────
echo ""
echo "── Phase 8: Validation metrics ──────────────────────────────────────────"
python src/04_analysis/validate.py --journal "$JOURNAL"

# ── Phase 9: Figures ──────────────────────────────────────────────────────────
echo ""
echo "── Phase 9: Generating interactive figures ──────────────────────────────"
python src/05_visualization/generate_figures.py --journal "$JOURNAL"

# ── Phase 10: Tables ──────────────────────────────────────────────────────────
echo ""
echo "── Phase 10: Generating publication tables ──────────────────────────────"
python src/05_visualization/generate_tables.py --journal "$JOURNAL"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Pipeline complete!  $(date)"
echo "=================================================="
echo ""
echo "Outputs:"
echo "  Figures   → outputs/figures/"
echo "  Tables    → outputs/tables/"
echo "  Reports   → outputs/reports/"
echo "  VOSviewer → outputs/vosviewer/"
echo "  Model     → outputs/models/bertopic_model_${JOURNAL}/"
echo ""
echo "Open any .html file in outputs/figures/ for interactive visualization."
