"""Verify Python version and all dependencies before running the pipeline."""
import sys


def check_python_version():
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major < 3 or (major == 3 and minor < 9):
        print(f"  ✗ Python {version_str} — requires Python 3.9 or higher")
        return False
    if major == 3 and minor == 9:
        print(f"  ⚠ Python {version_str} — use files/requirements_py39.txt (scipy==1.13.1)")
        return True
    print(f"  ✓ Python {version_str}")
    return True


def check_import(module, name=None):
    try:
        __import__(module)
        print(f"  ✓ {name or module}")
        return True
    except ImportError as e:
        print(f"  ✗ {name or module}: {e}")
        return False


def check_env():
    """Warn if .env is missing or still contains the placeholder email."""
    from pathlib import Path
    env_path = Path(".env")
    if not env_path.exists():
        print("  ✗ .env file missing — copy files/.env.template to .env and fill in PUBMED_EMAIL")
        return False
    content = env_path.read_text()
    if "your_email@institution.edu" in content:
        print("  ✗ .env contains placeholder email — replace with your real PubMed email")
        return False
    if "PUBMED_EMAIL=" not in content or "PUBMED_EMAIL=\n" in content:
        print("  ✗ PUBMED_EMAIL is empty in .env")
        return False
    print("  ✓ .env present and PUBMED_EMAIL is set")
    return True


checks = [
    ("bertopic", "BERTopic"),
    ("sentence_transformers", "SentenceTransformers"),
    ("umap", "UMAP"),
    ("hdbscan", "HDBSCAN"),
    ("safetensors", "safetensors"),
    ("Bio.Entrez", "Biopython/Entrez"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("plotly", "Plotly"),
    ("gensim", "Gensim"),
    ("sklearn", "Scikit-learn"),
    ("torch", "PyTorch"),
    ("scipy", "SciPy"),
    ("statsmodels", "Statsmodels"),
    ("nltk", "NLTK"),
    ("loguru", "Loguru"),
    ("dotenv", "python-dotenv"),
    ("tqdm", "tqdm"),
    ("openpyxl", "openpyxl"),
]

print("=== BERTopic Pipeline Setup Check ===\n")
print("Python version:")
py_ok = check_python_version()

print("\nDependencies:")
results = [check_import(m, n) for m, n in checks]

print("\nEnvironment:")
env_ok = check_env()

n_failed = results.count(False)
if py_ok and n_failed == 0 and env_ok:
    print("\n✓ All checks passed. Ready to run: bash run_pipeline.sh")
else:
    if not py_ok:
        print(f"\n✗ Python version check failed.")
    if n_failed:
        print(f"\n✗ {n_failed} missing package(s). Run: pip install -r files/requirements.txt")
    if not env_ok:
        print(f"\n✗ Environment setup incomplete. Fix .env before running Phase 1.")
    sys.exit(1)
