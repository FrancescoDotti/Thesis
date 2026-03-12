# Thesis

This folder contains the code, data, outputs, and writing files for the thesis project.

## Folder overview

- `Data/`: input datasets and cached raw data used by the analysis scripts and notebooks.
- `Outputs/`: generated figures and tables created by the analysis.
- `Scripts/`: Python scripts and Jupyter notebooks used to prepare data, run the decomposition, and produce results.
- `.vscode/`: local VS Code settings for LaTeX and editor behavior.
- `.venv/`: local Python virtual environment for this project.

## Main files

- `Bibliography.tex`: LaTeX source file for bibliography-related writing.
- `Bibliography.pdf`: compiled PDF corresponding to the bibliography document.
- `What moves stock prices.pdf`: reference paper used for the empirical framework.
- `Thesis.code-workspace`: VS Code workspace file for opening this project with the saved workspace settings.
- `.gitignore`: Git rules that keep source files and key inputs in the repository while excluding generated clutter.

## Data files

- `Data/BigSmall_NYA.xlsx`: main Excel dataset used by the thesis analysis notebooks and scripts.
- `Data/CRSP.csv`: cached CRSP daily data used for the WRDS-based decomposition workflow.

## Script files

- `Scripts/Thesis_2_2.py`: Python script for the variance decomposition analysis based on `BigSmall_NYA.xlsx`.
- `Scripts/Thesis_3_2 + Roll's r2.py`: Python script that computes the Brogaard-style decomposition and Roll's `R^2` outputs.
- `Scripts/download_nyse_daily_returns.py`: helper script that downloads and stores CRSP daily return data in `Data/`.
- `Scripts/Tesi_2_1.ipynb`: notebook version of the variance decomposition workflow.
- `Scripts/Thesis_3.ipynb`: notebook that loads `BigSmall_NYA.xlsx`, prepares the panel data, and computes decomposition results.
- `Scripts/Tesi_3_1.ipynb`: notebook for the WRDS/CRSP-based decomposition workflow.

## Output files

- `Outputs/roll_r2_results.csv`: main table of Roll's `R^2` results.
- `Outputs/roll_r2_by_year.csv`: yearly Roll's `R^2` summary.
- `Outputs/roll_r2_over_time.png`: figure showing how Roll's `R^2` evolves over time.
- `Outputs/r2_vs_noise_share.png`: figure comparing Roll's `R^2` with the estimated noise share.

## Working convention

Scripts and notebooks read inputs from `Data/` and write generated results to `Outputs/`. This keeps source material separate from reproducible outputs and makes the repository easier to manage on GitHub.
