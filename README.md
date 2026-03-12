# Thesis

This folder contains the code, data, outputs, and writing files for the thesis project.

## Folder overview

- `Data/`: input datasets and cached raw data used by the analysis scripts and notebooks.
- `Outputs/`: generated figures and tables created by the analysis.
- `Scripts/`: Python scripts and Jupyter notebooks used to prepare data, run the decomposition, and produce results.

## Main files

- `Bibliography.tex`: LaTeX source file for bibliography-related writing.
- `Bibliography.pdf`: compiled PDF corresponding to the bibliography document.
- `What moves stock prices.pdf`: reference paper used for the empirical framework.

## Data files

- `Data/BigSmall_NYA.xlsx`: main Excel dataset used by the thesis analysis notebooks and scripts.

## Script files

- `Scripts/Thesis_2_2.py`: Python script for the variance decomposition analysis based on `BigSmall_NYA.xlsx`.
- `Scripts/Tesi_2_1.ipynb`: notebook version of the variance decomposition workflow.
- `Scripts/Thesis_3.ipynb`: notebook that loads `BigSmall_NYA.xlsx`, prepares the panel data, and computes decomposition results.
- `Scripts/Tesi_3_1.ipynb`: notebook for the WRDS/CRSP-based decomposition workflow.

## Working convention

Scripts and notebooks read inputs from `Data/` and write generated results to `Outputs/`. This keeps source material separate from reproducible outputs and makes the repository easier to manage on GitHub.
