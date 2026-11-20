#!/usr/bin/env python3
"""Step 1: per-paper share of LLM-modified main-text sentences (alpha), old window.

Converted from the project notebook code/get_llm_detect_main.ipynb (cells: config, model load,
LaTeX main-text extraction, tokenizer, extraction loop, tokenisation, per-paper MLE, save).
Notebook cells for the corpus-level and monthly alpha (SI timeline figure) and test cells are omitted.

Estimator: distributional MLE of Liang et al. (2025, Nat. Hum. Behav. 9, "Quantifying large language
model usage in scientific papers"), third-party code, not included here:
  https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
  clone it and pass --liang-repo; we use its `src.MLE.MLE` and the physics reference distribution
  `distribution/Phys.parquet` (word frequencies of human-written vs LLM-generated physics text).
Main text = body between \\begin{document} and \\end{document}, without abstract, title/author,
appendix, bibliography and acknowledgements, LaTeX stripped; sentences split by spaCy
`en_core_web_sm`; papers with < 5 sentences get NaN. A paper is LLM-assisted if alpha > 0.1 (applied
in the pair frame and make_fig5.py).

Run (old window):
  python 01_alpha_main.py --metadata data/arxiv_physics_2024_metadata.csv \
      --sources data/arxiv_sources_physics --out data/arxiv_physics_2024_llm_scores_main.csv \
      --liang-repo ~/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
Output: the metadata columns + llm_alpha_main, llm_alpha_main_ci, has_main_text (28,853 papers with alpha).
The functions below are imported by 02_alpha_main_parallel.py (new window).
"""
import os
import sys
import re
import tarfile
import gzip
import io
import tempfile

import numpy as np


# ---------- LaTeX main-text extraction (notebook cell 7, unchanged) ----------

def extract_tex_from_archive(archive_path):
    """
    Extract .tex file contents from a .tar.gz archive.
    Uses gzip -> BytesIO -> tarfile method (confirmed working on Discovery).
    Also handles: single gzipped .tex, plain .tex, and zip archives.
    """
    tex_contents = {}

    # Method 1: gzip decompress, then open as tar (this is what works for our archives)
    try:
        with gzip.open(archive_path, 'rb') as gz:
            raw = io.BytesIO(gz.read())
        raw.seek(0)
        if tarfile.is_tarfile(raw):
            raw.seek(0)
            with tarfile.open(fileobj=raw) as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".tex") and member.isfile():
                        f = tar.extractfile(member)
                        if f:
                            try:
                                tex_contents[member.name] = f.read().decode("utf-8", errors="replace")
                            except:
                                pass
            if tex_contents:
                return tex_contents
        else:
            # It's gzipped but not a tar — probably a single .tex file
            raw.seek(0)
            content = raw.read().decode("utf-8", errors="replace")
            bd = "\\begin{document}"
            if bd in content or "\\section" in content:
                tex_contents["main.tex"] = content
                return tex_contents
    except:
        pass

    # Method 2: direct tarfile (works on some systems)
    try:
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".tex") and member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        try:
                            tex_contents[member.name] = f.read().decode("utf-8", errors="replace")
                        except:
                            pass
        if tex_contents:
            return tex_contents
    except:
        pass

    # Method 3: zip archive
    try:
        import zipfile
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith(".tex"):
                        try:
                            tex_contents[name] = zf.read(name).decode("utf-8", errors="replace")
                        except:
                            pass
        if tex_contents:
            return tex_contents
    except:
        pass

    # Method 4: plain text .tex file
    try:
        with open(archive_path, 'r', errors='replace') as f:
            content = f.read()
        bd = "\\begin{document}"
        if bd in content:
            tex_contents["main.tex"] = content
    except:
        pass

    return tex_contents


def find_main_tex(tex_contents):
    """Find the .tex file containing \\begin{document}."""
    bd = "\\begin{document}"
    for name, content in tex_contents.items():
        if bd in content:
            return name, content
    # Fallback: longest file
    if tex_contents:
        name = max(tex_contents, key=lambda k: len(tex_contents[k]))
        return name, tex_contents[name]
    return None, None


def strip_latex(text):
    """Remove LaTeX commands/environments, keep readable prose."""
    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Remove math/figure/table environments
    for env in ['equation', 'equation*', 'align', 'align*', 'eqnarray',
                'eqnarray*', 'figure', 'figure*', 'table', 'table*',
                'tikzpicture', 'lstlisting', 'verbatim', 'array',
                'tabular', 'tabular*', 'algorithmic', 'algorithm',
                'minted', 'displaymath', 'widetext', 'subequations']:
        pattern = r'\\begin\{' + re.escape(env) + r'\}.*?\\end\{' + re.escape(env) + r'\}'
        text = re.sub(pattern, ' ', text, flags=re.DOTALL)

    # Remove inline math
    text = re.sub(r'\$[^$]+\$', ' ', text)
    text = re.sub(r'\\\([^)]+\\\)', ' ', text)

    # Remove \cite, \ref, \label, \footnote
    text = re.sub(r'\\(?:cite|citep|citet|ref|eqref|label|autoref|cref|pageref)\*?(?:\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\footnote\{[^}]*\}', '', text)

    # Keep arguments of formatting commands
    for cmd in ['textbf', 'textit', 'emph', 'underline', 'textrm', 'textsc',
                'texttt', 'textsf', 'mathrm', 'mathbf']:
        text = re.sub(r'\\' + cmd + r'\{([^}]*)\}', r'\1', text)

    # Remove remaining commands
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*', ' ', text)

    # Remove braces
    text = re.sub(r'[{}]', '', text)

    # Clean whitespace
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract_main_text(tex_content):
    """
    Extract main body text (between \\begin{document} and \\end{document}),
    excluding the abstract, acknowledgments, bibliography, and appendix.
    """
    bd = "\\begin{document}"
    ed = "\\end{document}"

    # Get document body
    start = tex_content.find(bd)
    end = tex_content.find(ed)
    if start == -1:
        return None
    body = tex_content[start + len(bd):end if end != -1 else len(tex_content)]

    # Remove abstract
    body = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', '', body, flags=re.DOTALL)

    # Remove title/author/maketitle
    body = re.sub(r'\\maketitle', '', body)
    body = re.sub(r'\\title\{[^}]*\}', '', body)
    body = re.sub(r'\\author\{[^}]*\}', '', body)

    # Cut at appendix / bibliography / references
    for marker in [r'\\appendix', r'\\bibliography\{', r'\\begin\{thebibliography\}']:
        cut = re.search(marker, body)
        if cut:
            body = body[:cut.start()]

    # Remove acknowledgments section
    body = re.sub(
        r'\\section\*?\{[^}]*(acknowledgment|acknowledgement|acknowledgments|acknowledgements)[^}]*\}.*?(?=\\section|$)',
        '', body, flags=re.DOTALL | re.IGNORECASE
    )

    # Strip LaTeX commands
    clean = strip_latex(body)

    # Must have meaningful content
    if len(clean) < 200:
        return None

    return clean


def get_main_text_for_paper(arxiv_id, sources_dir):
    """Full pipeline: archive → tex → main text string."""
    if not isinstance(arxiv_id, str):
        return None

    fname = arxiv_id.replace("/", "_")
    candidates = [
        os.path.join(sources_dir, fname + ".tar.gz"),
        os.path.join(sources_dir, fname + ".gz"),
        os.path.join(sources_dir, fname),
    ]

    archive_path = None
    for c in candidates:
        if os.path.exists(c):
            archive_path = c
            break
    if archive_path is None:
        return None

    tex_contents = extract_tex_from_archive(archive_path)
    if not tex_contents:
        return None

    _, main_content = find_main_tex(tex_contents)
    if not main_content:
        return None

    return extract_main_text(main_content)


# ---------- Tokenizer (notebook cell 9; same as the abstract pipeline) ----------

def tokenize_text(text):
    """
    Tokenize text into sentences of word arrays.
    Identical to tokenize_abstract in get_llm_detect.ipynb.
    """
    text = text.replace('\n', ' ')
    doc = nlp(text)
    sentence_list = []
    for sent in doc.sents:
        words = re.findall(r'\b\w+\b', sent.text.lower())
        words_without_digits = [w for w in words if not w.isdigit()]
        if len(words_without_digits) != 0:
            sentence_list.append(np.array(words_without_digits))
    return sentence_list


def main():
    global nlp
    import argparse

    import pandas as pd
    import spacy
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="step-1 metadata CSV (old window)")
    ap.add_argument("--sources", required=True, help="directory of source archives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--liang-repo", default=os.path.expanduser("~/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers"))
    a = ap.parse_args()

    # --- PATHS (notebook cell 3) ---
    METADATA_CSV = a.metadata
    SOURCES_DIR = a.sources
    OUTPUT_CSV = a.out
    REPO_DIR = a.liang_repo
    DISTRIBUTION_PARQUET = os.path.join(REPO_DIR, "distribution", "Phys.parquet")
    TEMP_DIR = os.path.join(tempfile.gettempdir(), "llm_detect_main_temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    sys.path.insert(0, REPO_DIR)

    # --- Load MLE model & metadata (cell 5) ---
    from src.MLE import MLE

    model = MLE(DISTRIBUTION_PARQUET)
    print(f"MLE model loaded from {DISTRIBUTION_PARQUET}")

    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} papers")

    nlp = spacy.load("en_core_web_sm")
    df["arxiv_id"] = df["arxiv_id"].astype(str)   # set in notebook cell 11 (test cell, otherwise omitted)

    # --- Extract main text for all papers (cell 13) ---
    print("Extracting main text from LaTeX sources...")

    arxiv_ids = df["arxiv_id"].dropna().unique()

    # Check for existing progress
    already_done = set()
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        if "llm_alpha_main" in df_existing.columns:
            already_done = set(df_existing.loc[df_existing["llm_alpha_main"].notna(), "arxiv_id"])
        print(f"Already scored: {len(already_done)} papers")

    to_process = [aid for aid in arxiv_ids if aid not in already_done]
    print(f"Papers to process: {len(to_process)}")

    main_texts = {}
    n_success = 0
    n_fail = 0

    for aid in tqdm(to_process, desc="Extracting main text"):
        text = get_main_text_for_paper(aid, SOURCES_DIR)
        if text is not None:
            main_texts[aid] = text
            n_success += 1
        else:
            n_fail += 1

    print(f"\nExtraction results:")
    print(f"  Success: {n_success}")
    print(f"  Failed:  {n_fail} (no source archive, no .tex, or too short)")

    # --- Tokenize (cell 17, corpus-level part omitted) ---
    print("Tokenizing all main texts...")

    paper_sentences = {}
    for aid, text in tqdm(main_texts.items(), desc="Tokenizing"):
        paper_sentences[aid] = tokenize_text(text)

    total_sents = sum(len(s) for s in paper_sentences.values())
    print(f"Total sentences: {total_sents}")
    print(f"Avg per paper: {total_sents / max(len(paper_sentences), 1):.1f}")

    # --- Per-paper alpha (cell 23) ---
    print(f"Running per-paper inference on {len(paper_sentences)} papers...")

    alphas = {}
    cis_dict = {}
    temp_path = os.path.join(TEMP_DIR, "single_main.parquet")

    for aid in tqdm(sorted(paper_sentences.keys()), desc="Per-paper MLE (main)"):
        sents = paper_sentences[aid]
        if len(sents) < 5:  # main text should have more sentences than abstracts
            alphas[aid] = np.nan
            cis_dict[aid] = np.nan
            continue

        pd.DataFrame({"inference_sentence": sents}).to_parquet(temp_path)
        try:
            a_, c_ = model.inference(temp_path, exploded_data=True)
            alphas[aid] = a_
            cis_dict[aid] = c_
        except Exception:
            alphas[aid] = np.nan
            cis_dict[aid] = np.nan

    if os.path.isfile(temp_path):
        os.remove(temp_path)

    # --- Save (cell 25, monthly-corpus output omitted) ---
    out = df.copy()
    out["llm_alpha_main"] = out["arxiv_id"].map(alphas)
    out["llm_alpha_main_ci"] = out["arxiv_id"].map(cis_dict)
    out["has_main_text"] = out["arxiv_id"].isin(main_texts).astype(int)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
