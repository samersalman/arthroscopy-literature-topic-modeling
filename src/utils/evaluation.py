"""
Shared evaluation utilities for BERTopic pipeline.
Imported by run_bertopic.py, validate.py, and any future scripts that need
topic coherence or diversity metrics.
"""

import re
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize_for_coherence(text: str) -> list[str]:
    """Lowercase + keep only alphanum tokens — matches CountVectorizer defaults."""
    return _TOKEN_RE.findall(text.lower())


def compute_coherence(topic_model, docs: list[str], top_n: int = 10) -> dict:
    """Compute NPMI topic coherence using gensim."""
    try:
        topics = topic_model.get_topics()
        topic_words = [
            [word for word, _ in topics[t_id][:top_n]]
            for t_id in sorted(topics.keys())
            if t_id != -1
        ]
        tokenized = [tokenize_for_coherence(doc) for doc in docs]
        dictionary = Dictionary(tokenized)
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        score = coherence_model.get_coherence()
        return {"coherence_npmi": round(score, 4), "n_topics": len(topic_words)}
    except Exception as e:
        from loguru import logger
        logger.warning(f"Coherence computation failed: {e}")
        return {"coherence_npmi": None, "n_topics": len(topic_model.get_topics()) - 1}


def compute_diversity(topic_model, top_n: int = 10) -> float:
    """Compute topic diversity (proportion of unique words across all topics)."""
    try:
        topics = topic_model.get_topics()
        all_words = [
            word
            for t_id, words in topics.items()
            if t_id != -1
            for word, _ in words[:top_n]
        ]
        return round(len(set(all_words)) / len(all_words), 4) if all_words else 0.0
    except Exception:
        return 0.0
