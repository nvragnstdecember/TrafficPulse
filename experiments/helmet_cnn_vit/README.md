# CNN vs ViT — mandatory helmet-state experiment (P4-U5)

The project specification (`TRAFFICPULSE_MASTER_SPEC.md` §4) requires one controlled
CNN-versus-ViT comparison. `docs/architecture-review.md` §12 locks it to helmet-state
classification with **ResNet-50 vs DeiT-Small**; this directory executes it.

The design is frozen in [PREREGISTRATION.md](PREREGISTRATION.md), committed and git-tagged
**before** the first training run. Read that first — it is the contract this code implements,
including every deviation from §12 and what may and may not be claimed from the result.

## Attribution (required by the dataset licence)

> Siebert, F.W. & Lin, H. — HELMET dataset (OSF, <https://osf.io/4pwj8/>), CC-BY 4.0.

The dataset is CC-BY-4.0, verified 2026-08-29 from the OSF API (licence object
`563c1cf88c5e4a3877f9e96a`). Attribution is mandatory in any artifact derived from it.
**Nothing from the dataset is committed to this repository** — `data/` and `runs/` are
gitignored, and no model weights are vendored.

## Layout

| Module | Role | Heavy deps |
|---|---|---|
| `labels.py` | The 36-class positional-encoding grammar (`DNoHelmetP1NoHelmet` → driver state) | none |
| `corpus.py` | Annotation CSVs → sampled `CropRecord`s under the frozen policy | none |
| `official_split.py` | The authors' `data_split.csv`, applied verbatim and leakage-re-validated | none |
| `extract.py` | Crop harvesting, streamed straight out of the `part_N.zip` archives | Pillow (lazy) |
| `datasets.py` | Crop dataset + the per-family augmentation recipes | torch/timm (lazy) |
| `models.py` | The locked §12 model pair, behind one `create_model` | torch/timm (lazy) |
| `train.py` | One training loop, used by both families | torch (lazy) |
| `sweep.py` | The pre-registered tuning grid and final seed runs | torch (lazy) |
| `metrics.py` | macro-F1, per-class P/R/F1, PR-AUC, confusion, ECE | numpy |
| `calibrate.py` | Temperature scaling, fitted on validation only | numpy |
| `stats.py` | McNemar + paired bootstrap + the pre-committed decision rule | numpy |
| `robustness.py` | Corruptions and slice bucketing | Pillow (lazy) |
| `corrupt_eval.py` | The corruption pass (needs re-inference, so opt-in) | torch (lazy) |
| `bench.py` | Latency / throughput / VRAM, to the §12 protocol | torch (lazy) |
| `evaluate.py`, `report.py` | Assembly into `results.json` + `summary.md` | none |

Nothing here ships in the `trafficpulse` wheel. The harness tests
(`tests/experiments/test_cnnvit_*.py`, `test_helmet_experiment_harness.py`) run in CI **without**
torch or timm, via the same `find_spec` gating the H4/H5 tests use.

## Environment

The experiment needs a CUDA torch build, which the application environment does not. Keep it in
a **separate** virtualenv so the app's `.venv` is untouched:

```bash
py -3.13 -m venv .venv-cnnvit
PIP_CACHE_DIR=D:/pip-cache .venv-cnnvit/Scripts/python.exe -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 torch torchvision
.venv-cnnvit/Scripts/python.exe -m pip install timm pydantic
```

## Runbook

```bash
export PYTHONPATH=$PWD/experiments

# 1. Acquire (once). The licence/access gate must be resolved in
#    registry/datasets/helmet-myanmar.yaml first; see dataset-policy §14.
#    Small files first: Readme.txt=pvecy, data_split.csv=q7rmb, annotation.zip=buh57,
#    then image/part_1..7.zip. Verify every byte size against the OSF API listing.

# 2. Corpus + official split + crops (~40k crops, ~0.6 GB, a few minutes)
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit prepare

# 3. The pre-registered sweep: 3 LRs x 2 families x 6 epochs, then 3 seeds x 12 epochs
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit sweep --num-workers 6

# 4. Metrics, statistics, cost table, and the verdict
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit evaluate --bench
```

`prepare` refuses to continue if any clip is missing from the archives or if the crop count
does not match the frozen manifest — a partial extraction must never be trained on.

## Operational notes

Two things cost real time on this machine and are worth knowing:

- **Worker count dominates throughput.** The loop is data-starved, not GPU-bound, below about
  six workers: measured 79 img/s at `--num-workers 0`, 433 at 2, 859 at 6, and 674 at 8
  (oversubscribed). The GPU itself sustains ~324 img/s for ResNet-50 and ~426 for DeiT-Small,
  so six workers is where training becomes GPU-bound. Workers are spent only on the training
  loader; evaluation runs in the main process, because three loaders' worth of persistent
  workers exhausted a 16 GB machine.
- **Frame ids inside the archives are zero-padded to two digits** (`01.jpg`, not `1.jpg`).
  `extract_crops` counts every member it cannot open rather than skipping it, which is how
  that was caught instead of silently losing the crops from frames 1–9.
