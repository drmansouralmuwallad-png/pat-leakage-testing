# PAT: black-box testing of machine-learning feature pipelines for information leakage

This repository contains the perturbation admissibility test (PAT) and the full replication package of the paper
*Black-Box Testing of Machine-Learning Feature Pipelines for Information Leakage* (M. Almuwallad, submitted to *Information and Software Technology*).

PAT reruns an **unmodified** feature-pipeline script on copies of the data in which only information that is unavailable at prediction time (future values, test labels) has been perturbed. Any learner-facing column that changes depends on unavailable information, so it is reported as leakage. PAT needs no trained model, no test oracle and no access to the pipeline's internals.

## Use PAT on your own pipeline
Your script must read its data with `pandas.read_csv` and leave `X_train`, `X_test`, `y_train` and `y_test` in its global namespace, as a notebook would.

```python
from pat_core import ScriptPipeline, pat_temporal, determinism_check   # code/pat_core.py
pipe = ScriptPipeline('features.py')            # your script, unchanged
assert not determinism_check(pipe, df)          # fix seeds first if this lists columns
res = pat_temporal(pipe, df, audit_rows=[len(df) - 56, len(df) - 1], B=5, scale=2.0)
print(res['flagged'], [a['flagged'] for a in res['audits']])
```
`code/examples/test_pipeline_leakage.py` shows PAT as a pytest test that can run in continuous integration. For tabular data use `pat_tabular`.

## Replication package
The package reproduces every table, figure and number in the manuscript and its supplement.

## Layout
| Folder | Contents |
|---|---|
| `code/pat_core.py` | PAT: runs unmodified pipeline scripts, temporal and tabular information models, determinism pre-check |
| `code/examples/` | PAT as a pytest test for continuous integration (`pytest -q code/examples`) |
| `code/corpus_temporal.py`, `corpus_tabular.py`, `corpus_library.py` | clean pipelines, the 26 leakage mutation operators, library cases; render the scripts in `sources/` |
| `code/data_temporal.py`, `data_tabular.py` | series loading and reporting-gap mask; tabular datasets (sampling, 5% missing values) |
| `code/run_temporal.py`, `run_tabular.py` | PAT, shift check, sensitivity grid and equivalence oracle on every instance |
| `code/static_baselines.py` | time-offset check and the static analyser of Yang et al. (ASE'22) |
| `code/run_deepchecks.py` | Deepchecks baseline (run in its own environment) |
| `code/run_library.py` | RQ4 library-transformer audit |
| `code/case_study.py` | RQ5 case study (four versions of the digital-twin pipeline) |
| `code/scalability.py`, `oracle_detail.py`, `analyse_misses.py`, `make_graphical_abstract.py` | RQ3 cost; oracle diagnostics; misses of the controlled benchmark |
| `code/analyse.py` | computes every reported quantity into `results/summary.json` |
| `code/make_tables.py`, `make_figs.py` | write the LaTeX tables and figures from `summary.json` |
| `sources/` | all 238 pipeline scripts (104 temporal, 48 tabular, 82 library, 4 case study) and their indices |
| `data/` | `panels.npy` (the 150 series), the three tabular datasets as used, `tabular_meta.json` |
| `results/` | per-instance results (`*.jsonl`), static and Deepchecks verdicts, `summary.json` |
| `legacy/` | the controlled-generator benchmark and earlier case-study code and results (Supplementary Tables S5-S11) |

## Quick check (about 30 seconds)
```
bash verify_package.sh          # or: PYTHON=/path/to/python bash verify_package.sh
```
It runs the PAT unit tests, regenerates `summary.json` from the stored results and checks that
it is identical, and re-runs one temporal series (26 pipelines) and one split of each tabular
dataset (48 pipelines), comparing every audit with the stored records.

## Reproduce
```
pip install -r requirements.txt
python code/corpus_temporal.py sources            # optional: re-render scripts
python code/run_temporal.py sources data/panels.npy results/temporal.jsonl 2 dump/temporal
python code/run_tabular.py  sources data/tabular_meta.json results/tabular.jsonl 10 dump/tabular
python code/run_library.py  sources data/panels.npy data/tabular_meta.json results/library.jsonl
python code/static_baselines.py sources results/static.json     # needs the analyser (see requirements.txt)
<deepchecks-python> code/run_deepchecks.py dump/temporal results/deepchecks_temporal.jsonl regression results/dc_temporal_files.txt
<deepchecks-python> code/run_deepchecks.py dump/tabular  results/deepchecks_tabular.jsonl binary
python code/case_study.py data/panels.npy results/case_study.json
python code/scalability.py scal results/scalability.json
python code/oracle_detail.py results
python code/analyse.py results
python code/make_tables.py results <latex-dir>; python code/make_figs.py results <latex-dir>/figs
```
To reproduce only the reported numbers from the stored results, run the last three commands.
Runs are deterministic: seeds are derived from instance identifiers (CRC32). Re-running one
series reproduces the stored audit records exactly. Timings depend on hardware: the study
used 2 vCPU (Intel Xeon, 2.10 GHz) with about 8 GB of memory, one core per run.

The static analyser is located through the environment variables `LEAKAGE_ANALYSIS_DIR`
and `LEAKAGE_ANALYSIS_PYTHON`.

## Data sources
M5 (Makridakis et al., 2022), M4 (Makridakis et al., 2020), UCI Electricity Load Diagrams
(Trindade, 2015; hourly version of Lai et al., 2018); UCI Adult (Becker and Kohavi, 1996),
Statlog German Credit (Hofmann, 1994), Breast Cancer Wisconsin Diagnostic (Wolberg et al.,
1995). Adult and German Credit were obtained from the mirror github.com/jbrownlee/Datasets
(SHA-256 of the raw files in `data/SHA256SUMS`).

## Licence
MIT (see LICENSE).

## Citation
If you use this code, please cite the paper and the archived release (see `CITATION.cff`). The Zenodo DOI will be added here after the first release.
