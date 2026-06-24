"""
Optional Phase 6b: LLM Topic Label Refinement
===============================================
Uses OpenAI GPT-4o to generate concise clinical labels for each BERTopic topic.
Run AFTER run_bertopic.py. Labels are written back into topic_info and
temporal_trends CSVs so all downstream scripts use them automatically.

Requirements:
    pip install openai==1.40.0
    OPENAI_API_KEY set in .env

Usage:
    python src/03_modeling/label_topics_llm.py --journal ARTHROSCOPY
    python src/03_modeling/label_topics_llm.py --journal ARTHROSCOPY --dry_run

Outputs:
    outputs/tables/topic_labels_llm_ARTHROSCOPY.json
    outputs/tables/topic_info_ARTHROSCOPY.csv      (llm_label column added)
"""

import argparse
import json
import time
from pathlib import Path

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

import pandas as pd
from bertopic import BERTopic
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROMPT_TEMPLATE = """You are an expert sports medicine orthopedic surgeon reviewing research topics.

The following words describe a cluster of published research abstracts from the journal Arthroscopy:
{words}

Generate a concise clinical label (4-6 words maximum) that accurately describes
the primary research theme. Respond with ONLY the label, no explanation.

Examples of good labels:
- "ACL reconstruction graft outcomes"
- "Rotator cuff repair techniques"
- "Return to sport after knee injury"
- "Hip arthroscopy FAI treatment"
- "Shoulder instability surgical repair"
"""


def label_topics(journal: str = "ARTHROSCOPY", dry_run: bool = False):
    try:
        from openai import OpenAI
    except ImportError:
        logger.error(
            "openai package not installed. Run: pip install openai==1.40.0"
        )
        raise

    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        logger.error("OPENAI_API_KEY not set in .env — cannot run LLM labeling")
        raise ValueError("Set OPENAI_API_KEY in .env first")

    table_dir = Path("outputs/tables")
    model_path = f"outputs/models/bertopic_model_{journal}"
    topic_info_path = table_dir / f"topic_info_{journal}.csv"

    topic_model = BERTopic.load(model_path)
    topic_info = pd.read_csv(topic_info_path)

    if dry_run:
        logger.info("DRY RUN — showing prompts without calling API")
    else:
        client = OpenAI(api_key=api_key)

    labels = {}
    non_noise = topic_info[topic_info["Topic"] != -1]
    logger.info(f"Labeling {len(non_noise)} topics for {journal}...")

    for _, row in non_noise.iterrows():
        topic_id = row["Topic"]
        top_words = topic_model.get_topic(topic_id)
        words_str = ", ".join([w for w, _ in top_words[:15]])

        if dry_run:
            logger.info(f"  Topic {topic_id}: [{words_str}]")
            labels[topic_id] = f"[DRY RUN] Topic {topic_id}"
            continue

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(words=words_str)
                }],
                max_tokens=30,
                temperature=0.2,
            )
            label = response.choices[0].message.content.strip()
            labels[topic_id] = label
            logger.info(f"  Topic {topic_id}: {label}")
            time.sleep(0.2)  # avoid rate limit
        except Exception as e:
            logger.warning(f"  Topic {topic_id} failed: {e} — using BERTopic name")
            labels[topic_id] = row["Name"]

    labels[-1] = "Noise / Uncategorized"

    # Save JSON
    labels_path = table_dir / f"topic_labels_llm_{journal}.json"
    labels_path.write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2))
    logger.success(f"Labels saved: {labels_path}")

    # Write back into topic_info CSV
    topic_info["llm_label"] = topic_info["Topic"].map(labels)
    topic_info.to_csv(topic_info_path, index=False)

    # Write back into temporal_trends if it exists
    trends_path = table_dir / f"temporal_trends_{journal}.csv"
    if trends_path.exists():
        df_trends = pd.read_csv(trends_path)
        df_trends["llm_label"] = df_trends["topic_id"].map(labels)
        df_trends.to_csv(trends_path, index=False)

    logger.success(f"LLM labels written back to topic_info and temporal_trends CSVs.")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM topic labels with GPT-4o")
    parser.add_argument("--journal", default="ARTHROSCOPY")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print prompts without calling the API")
    args = parser.parse_args()
    label_topics(args.journal, args.dry_run)
