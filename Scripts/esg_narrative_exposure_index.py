"""
Build a Hassan-style ESG narrative exposure index from earnings-call transcripts.

Method idea (adapted from Hassan et al., 2019):
1) Parse each call transcript.
2) Detect ESG-related language (E, S, G dictionaries).
3) Detect risk language in the same sentence.
4) Use ESG-risk co-occurrence intensity as narrative exposure.

The script is intentionally simple and transparent for thesis replication.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'pypdf'. Install it with: pip install pypdf"
    ) from exc


# Resolve paths from the Thesis project root.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_DIR / "Data" / "Earning Calls_test"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Outputs" / "esg_narrative_test"


def resolve_input_dir(input_dir: Path) -> Path:
    """Resolve input directory, with fallback auto-detection under Data/."""
    # First, trust the user-provided or default path if it exists.
    if input_dir.exists():
        return input_dir

    # If it does not exist, search for likely earnings-call folders.
    data_dir = PROJECT_DIR / "Data"
    candidates: List[Path] = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "earning" in name and "call" in name:
            candidates.append(child)

    # Keep only candidates that actually contain at least one PDF.
    candidates = [folder for folder in candidates if any(folder.glob("*.pdf"))]

    if len(candidates) == 1:
        return candidates[0]

    # If no unique fallback is found, return original path and fail downstream with clear error.
    return input_dir


# ESG dictionaries (seed terms; easy to expand later).
ESG_DICTIONARY: Dict[str, List[str]] = {
    "E": [
        "climate", "emission", "emissions", "carbon", "decarbon", "energy",
        "renewable", "sustainable", "sustainability", "pollution", "biodiversity",
        "waste", "water", "green", "net zero", "transition"
    ],
    "S": [
        "employee", "employees", "labor", "workforce", "diversity", "inclusion",
        "community", "customer", "customers", "health", "safety", "training",
        "human rights", "wellbeing", "social", "welfare"
    ],
    "G": [
        "governance", "board", "compliance", "ethics", "control", "controls",
        "audit", "transparency", "accountability", "regulation", "regulatory",
        "supervision", "oversight", "conduct", "risk management"
    ],
}


# Risk words used to proxy risk-focused narrative context.
RISK_TERMS: List[str] = [
    "risk", "risks", "risky", "uncertain", "uncertainty", "volatile", "volatility",
    "exposure", "threat", "downside", "vulnerable", "pressure", "stressed", "shock"
]


@dataclass
class CallResult:
    """Container for one call-level ESG narrative result."""

    file_name: str
    total_words: int
    esg_mentions_E: int
    esg_mentions_S: int
    esg_mentions_G: int
    risk_mentions: int
    overlap_score_E: int
    overlap_score_S: int
    overlap_score_G: int
    nexp_E_per_1k: float
    nexp_S_per_1k: float
    nexp_G_per_1k: float
    nexp_total_per_1k: float


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file page by page."""
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize text for robust keyword matching."""
    text = text.lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9\s\.!\?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """Split text into coarse sentences for local co-occurrence checks."""
    chunks = re.split(r"[\.!\?]+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    """Count keyword mentions in a text block using regex word boundaries."""
    count = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        count += len(re.findall(pattern, text))
    return count


def compute_call_index(file_path: Path) -> Tuple[CallResult, pd.DataFrame]:
    """Compute call-level ESG narrative exposure and return diagnostics."""
    raw_text = extract_pdf_text(file_path)
    text = clean_text(raw_text)
    sentences = split_sentences(text)
    total_words = max(len(text.split()), 1)

    # Count document-level ESG and risk mentions for diagnostics.
    esg_mentions = {
        pillar: count_keyword_hits(text, terms)
        for pillar, terms in ESG_DICTIONARY.items()
    }
    risk_mentions = count_keyword_hits(text, RISK_TERMS)

    # Hassan-style adaptation: score overlap intensity in local narrative units (sentences).
    overlap_scores = {"E": 0, "S": 0, "G": 0}
    evidence_rows: List[Dict[str, object]] = []

    for sent in sentences:
        sent_risk = count_keyword_hits(sent, RISK_TERMS)
        if sent_risk == 0:
            continue

        for pillar, terms in ESG_DICTIONARY.items():
            sent_esg = count_keyword_hits(sent, terms)
            if sent_esg == 0:
                continue

            # Overlap intensity: ESG mentions weighted by concurrent risk mentions.
            overlap = sent_esg * sent_risk
            overlap_scores[pillar] += overlap

            evidence_rows.append(
                {
                    "file_name": file_path.name,
                    "pillar": pillar,
                    "esg_hits_in_sentence": sent_esg,
                    "risk_hits_in_sentence": sent_risk,
                    "overlap": overlap,
                    "sentence_excerpt": sent[:240],
                }
            )

    # Convert overlap to per-1,000-words exposure to normalize across transcript length.
    nexp_e = overlap_scores["E"] / total_words * 1000
    nexp_s = overlap_scores["S"] / total_words * 1000
    nexp_g = overlap_scores["G"] / total_words * 1000
    nexp_total = nexp_e + nexp_s + nexp_g

    call_result = CallResult(
        file_name=file_path.name,
        total_words=total_words,
        esg_mentions_E=esg_mentions["E"],
        esg_mentions_S=esg_mentions["S"],
        esg_mentions_G=esg_mentions["G"],
        risk_mentions=risk_mentions,
        overlap_score_E=overlap_scores["E"],
        overlap_score_S=overlap_scores["S"],
        overlap_score_G=overlap_scores["G"],
        nexp_E_per_1k=nexp_e,
        nexp_S_per_1k=nexp_s,
        nexp_G_per_1k=nexp_g,
        nexp_total_per_1k=nexp_total,
    )

    evidence_df = pd.DataFrame(evidence_rows)
    return call_result, evidence_df


def add_cross_call_standardization(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored index columns across calls for comparability."""
    output = results_df.copy()
    for col in ["nexp_E_per_1k", "nexp_S_per_1k", "nexp_G_per_1k", "nexp_total_per_1k"]:
        std = output[col].std(ddof=0)
        if std == 0:
            output[col.replace("per_1k", "z")]= 0.0
        else:
            output[col.replace("per_1k", "z")] = (output[col] - output[col].mean()) / std
    return output


def run_pipeline(input_dir: Path, output_dir: Path) -> None:
    """Run the full ESG narrative exposure pipeline and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    call_rows: List[Dict[str, object]] = []
    evidence_frames: List[pd.DataFrame] = []

    for pdf_file in pdf_files:
        call_result, evidence_df = compute_call_index(pdf_file)
        call_rows.append(call_result.__dict__)
        if not evidence_df.empty:
            evidence_frames.append(evidence_df)

    results_df = pd.DataFrame(call_rows)
    results_df = add_cross_call_standardization(results_df)

    # Save the main table with the index values.
    results_path = output_dir / "esg_narrative_exposure_index.csv"
    results_df.to_csv(results_path, index=False)

    # Save narrative evidence rows to inspect what is driving the index.
    evidence_path = output_dir / "esg_narrative_sentence_evidence.csv"
    if evidence_frames:
        pd.concat(evidence_frames, ignore_index=True).to_csv(evidence_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "file_name", "pillar", "esg_hits_in_sentence", "risk_hits_in_sentence",
                "overlap", "sentence_excerpt"
            ]
        ).to_csv(evidence_path, index=False)

    # Save a short markdown report for quick interpretation.
    report_path = output_dir / "esg_narrative_report.md"
    ranked = results_df.sort_values("nexp_total_per_1k", ascending=False)
    lines = [
        "# ESG Narrative Exposure Index (Hassan-style adaptation)",
        "",
        "This report uses sentence-level ESG × risk co-occurrence intensity, normalized per 1,000 words.",
        "",
        "## Calls ranked by total ESG narrative exposure",
        "",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"- {row['file_name']}: total={row['nexp_total_per_1k']:.3f}, "
            f"E={row['nexp_E_per_1k']:.3f}, S={row['nexp_S_per_1k']:.3f}, G={row['nexp_G_per_1k']:.3f}"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved index table: {results_path}")
    print(f"Saved sentence evidence: {evidence_path}")
    print(f"Saved quick report: {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a Hassan-style ESG narrative exposure index from earnings-call PDFs."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir)