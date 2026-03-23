"""
Build an ESG narrative exposure index from earnings-call transcripts using FinBERT-ESG.

Model: yiyanghkust/finbert-esg
Task: sentence-level ESG classification (Environmental, Social, Governance, None)

This script:
1) Reads PDF transcripts from an input folder.
2) Splits each transcript into sentences.
3) Runs FinBERT-ESG on each sentence.
4) Aggregates sentence scores into call-level exposure indices.
5) Saves outputs for analysis and diagnostics.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pypdf import PdfReader
from transformers import pipeline


# Resolve paths from the Thesis project root.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "Data" / "Earning Calls_test"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Outputs" / "finbert_esg_test"
MODEL_NAME = "yiyanghkust/finbert-esg"


def resolve_input_dir(input_dir: Path) -> Path:
    """Resolve input directory, with fallback auto-detection under Data/."""
    # First, use the provided path if it exists.
    if input_dir.exists():
        return input_dir

    # If missing, search for folder names that look like earnings-call data.
    data_dir = PROJECT_DIR / "Data"
    candidates: List[Path] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "earning" in name and "call" in name:
            candidates.append(child)

    # Keep only folders that actually contain PDF files.
    candidates = [folder for folder in candidates if any(folder.glob("*.pdf"))]

    # If there is exactly one clear match, use it.
    if len(candidates) == 1:
        return candidates[0]

    # Otherwise return the original path and let the caller raise a clear error.
    return input_dir


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file page by page."""
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize text to improve sentence splitting and model stability."""
    # Lowercasing keeps preprocessing consistent with the existing pipeline.
    text = text.lower()

    # Replace curly apostrophes that frequently appear in PDF extraction.
    text = text.replace("\u2019", "'")

    # Keep letters, numbers, and sentence punctuation.
    text = re.sub(r"[^a-z0-9\s\.!\?]", " ", text)

    # Collapse repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str, min_words: int = 5) -> List[str]:
    """Split text into coarse sentences and drop very short fragments."""
    chunks = re.split(r"[\.!\?]+", text)
    sentences = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Filter out tiny fragments that are usually table artifacts in PDF text.
    return [sent for sent in sentences if len(sent.split()) >= min_words]


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into small batches for model inference."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def add_cross_call_standardization(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored columns across calls for comparability."""
    output = results_df.copy()
    columns_to_scale = [
        "finbert_E_per_1k",
        "finbert_S_per_1k",
        "finbert_G_per_1k",
        "finbert_total_per_1k",
    ]

    for col in columns_to_scale:
        std = output[col].std(ddof=0)
        z_col = col.replace("per_1k", "z")
        if std == 0:
            output[z_col] = 0.0
        else:
            output[z_col] = (output[col] - output[col].mean()) / std

    return output


def run_pipeline(input_dir: Path, output_dir: Path, batch_size: int = 32) -> None:
    """Run the full FinBERT-ESG pipeline and save outputs."""
    # Resolve user path with safe fallback.
    input_dir = resolve_input_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files.
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    # Load classifier once to reuse for all transcripts.
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=512,
    )

    # Store per-call summaries and sentence-level diagnostics.
    call_rows: List[Dict[str, object]] = []
    sentence_rows: List[Dict[str, object]] = []

    # Process each PDF file independently.
    for pdf_file in pdf_files:
        raw_text = extract_pdf_text(pdf_file)
        text = clean_text(raw_text)
        sentences = split_sentences(text)
        total_words = max(len(text.split()), 1)

        # Initialize weighted and count metrics by label.
        weighted_scores = {
            "Environmental": 0.0,
            "Social": 0.0,
            "Governance": 0.0,
            "None": 0.0,
        }
        label_counts = {
            "Environmental": 0,
            "Social": 0,
            "Governance": 0,
            "None": 0,
        }

        # Run model in batches for speed and stability.
        for batch in chunk_list(sentences, batch_size):
            predictions = classifier(batch)

            # Store detailed diagnostics and update aggregates.
            for sent, pred in zip(batch, predictions):
                label = pred["label"]
                score = float(pred["score"])

                # Keep one row per sentence for transparency.
                sentence_rows.append(
                    {
                        "file_name": pdf_file.name,
                        "sentence": sent[:300],
                        "predicted_label": label,
                        "confidence": score,
                    }
                )

                # Sum confidence by predicted class (soft narrative exposure).
                weighted_scores[label] += score
                label_counts[label] += 1

        # Normalize exposure by transcript length (per 1,000 words).
        e_per_1k = weighted_scores["Environmental"] / total_words * 1000
        s_per_1k = weighted_scores["Social"] / total_words * 1000
        g_per_1k = weighted_scores["Governance"] / total_words * 1000
        total_per_1k = e_per_1k + s_per_1k + g_per_1k

        # Save call-level summary row.
        call_rows.append(
            {
                "file_name": pdf_file.name,
                "total_words": total_words,
                "num_sentences_scored": len(sentences),
                "count_environmental": label_counts["Environmental"],
                "count_social": label_counts["Social"],
                "count_governance": label_counts["Governance"],
                "count_none": label_counts["None"],
                "score_environmental": weighted_scores["Environmental"],
                "score_social": weighted_scores["Social"],
                "score_governance": weighted_scores["Governance"],
                "score_none": weighted_scores["None"],
                "finbert_E_per_1k": e_per_1k,
                "finbert_S_per_1k": s_per_1k,
                "finbert_G_per_1k": g_per_1k,
                "finbert_total_per_1k": total_per_1k,
            }
        )

    # Build output DataFrames.
    results_df = pd.DataFrame(call_rows)
    results_df = add_cross_call_standardization(results_df)
    sentence_df = pd.DataFrame(sentence_rows)

    # Save call-level table.
    results_path = output_dir / "finbert_esg_exposure_index.csv"
    results_df.to_csv(results_path, index=False)

    # Save sentence-level diagnostics.
    sentence_path = output_dir / "finbert_esg_sentence_predictions.csv"
    sentence_df.to_csv(sentence_path, index=False)

    # Save compact markdown report.
    report_path = output_dir / "finbert_esg_report.md"
    ranked = results_df.sort_values("finbert_total_per_1k", ascending=False)

    lines = [
        "# FinBERT-ESG Exposure Index",
        "",
        f"Model: `{MODEL_NAME}`",
        "",
        "Exposure is built from sentence-level class confidence sums (E+S+G), normalized per 1,000 words.",
        "",
        "## Calls ranked by total FinBERT ESG exposure",
        "",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            f"- {row['file_name']}: total={row['finbert_total_per_1k']:.3f}, "
            f"E={row['finbert_E_per_1k']:.3f}, S={row['finbert_S_per_1k']:.3f}, "
            f"G={row['finbert_G_per_1k']:.3f}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Print output locations for quick verification.
    print(f"Saved index table: {results_path}")
    print(f"Saved sentence predictions: {sentence_path}")
    print(f"Saved quick report: {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a FinBERT-ESG exposure index from earnings-call PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing earnings-call PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where output CSV/MD files will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sentences per model inference batch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir, batch_size=args.batch_size)
