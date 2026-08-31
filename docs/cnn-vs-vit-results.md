# CNN vs ViT for motorcycle driver helmet-state classification (P4-U5)

- **Status:** Complete — executed and reported
- **Date:** 2026-08-31
- **Pre-registration:** [`experiments/helmet_cnn_vit/PREREGISTRATION.md`](../experiments/helmet_cnn_vit/PREREGISTRATION.md), frozen at git tag `p4u5-prereg`
- **Source of truth:** `runs/helmet_cnn_vit/report/results.json` (gitignored; regenerate with the commands in [§10](#10-reproduction))
- **Authority:** `TRAFFICPULSE_MASTER_SPEC.md` §4; [`architecture-review.md`](architecture-review.md) §12; [`evaluation-protocol.md`](evaluation-protocol.md) §8; [`phase-4-plan.md`](phase-4-plan.md) P4-U5
- **Decision record:** [ADR-005](adr/ADR-005.md)

This document reports the outcome of the mandatory CNN-versus-ViT comparison required by
master spec §4. The design was frozen and git-tagged **before** the final seed runs; the
decision rule was committed in code (`helmet_cnn_vit.stats.decide`) before any test-split
number existed. Nothing below was chosen after seeing a result.

---

## 1. Headline

**ResNet-50 wins the accuracy comparison under the frozen decision rule.**

| | DeiT-Small (ViT) | ResNet-50 (CNN) |
|---|---|---|
| Mean test macro-F1 (3 seeds) | 0.91975 | **0.92881** |
| Std. dev. across seeds | 0.00261 | 0.00146 |
| Δ macro-F1 (DeiT − ResNet) | **−0.00906** | |
| Pooled bootstrap 95% CI | **[−0.01380, −0.00434]** | excludes zero |
| Sign-consistent across all 3 seeds | **yes** (all negative) | |

Both pre-registered conditions hold, so a difference is claimed. The margin is small —
roughly **0.9 macro-F1 points** — and the rule contains no minimum-effect threshold, so
the claim is reported exactly as the rule produces it, neither inflated nor discounted.

**The cost comparison points the other way.** On the measured RTX 4060 Laptop benchmark,
DeiT-Small is smaller and more efficient on every axis: fewer parameters, a smaller
checkpoint, lower latency, higher throughput, and lower peak VRAM — by ~30% latency and
~34% VRAM at batch 32. See [§7](#7-cost-benchmark).

So the honest summary is a **trade-off, not a sweep**: ResNet-50 is more accurate on this
task and this data; DeiT-Small is cheaper to run. Neither result licenses a deployment
decision on its own — see [§9](#9-limitations-and-anomalies).

---

## 2. Task, data, and splits

Binary classification of **driver** helmet state (`helmet` vs `no_helmet`) from the crop
of one annotated motorcycle. The target is the `D` token of HELMET's positional-encoding
label; passenger states are recorded as covariates only.

| | |
|---|---|
| Dataset | HELMET (Siebert & Lin), OSF node `4pwj8`, **CC-BY-4.0** |
| Attribution (required) | Siebert, F.W. and Lin, H. — HELMET dataset (OSF, <https://osf.io/4pwj8/>), CC-BY 4.0 |
| Corpus | 39,965 crops from 10,006 tracks; `corpus_hash` `ad42119c…c11147` |
| Split | The authors' official video-level `data_split.csv`, applied verbatim |
| Split manifest sha256 | `ce53d29b8a53d7be007a485e023e9d3875a882db6b0cd486f565dbe5c558b626` |

| Split | Videos | Crops | helmet | no_helmet | no_helmet share |
|---|---|---|---|---|---|
| train | 636 | 28,470 | 20,165 | 8,305 | 29.17% |
| val | 92 | 4,089 | 2,941 | 1,148 | 28.08% |
| test | 182 | 7,406 | 5,315 | 2,091 | 28.23% |

Leakage was re-validated independently with H3's `SplitValidator` under
`VideoAwareGrouping`: no video, track, or frame straddles a split. The test split was
read only after training completed, once per run.

**Class imbalance** is handled by class-weighted cross-entropy computed from **train**
counts. The test split was never rebalanced.

---

## 3. Model selection (validation only)

Equal budget per family: three learning rates × 6 epochs, selected on **validation**
macro-F1, then the selected configuration retrained for 12 epochs at seeds 0, 1, 2.

| LR | DeiT-Small val macro-F1 | ResNet-50 val macro-F1 |
|---|---|---|
| 1e-4 | **0.91714** ← selected | 0.90263 |
| 3e-4 | 0.89722 | 0.91383 |
| 1e-3 | 0.65236 | **0.92231** ← selected |

Each family's selected LR is the argmax of its own validation grid. **Both selections
landed on opposite boundaries of the frozen grid** — see [§9.1](#91-both-selected-learning-rates-sit-on-opposite-grid-boundaries).

---

## 4. Test-set results

All six final runs, evaluated on the same 7,406 held-out test crops. The six prediction
files carry identical crop ids in identical order and identical ground truth, so the
comparison is genuinely paired.

| Model | Seed | Test macro-F1 | Balanced acc. | Accuracy | PR-AUC (`no_helmet`) |
|---|---|---|---|---|---|
| DeiT-Small | 0 | 0.9216858754882578 | 0.916317 | 0.937348 | 0.932095 |
| DeiT-Small | 1 | 0.9207795033074323 | 0.919438 | 0.935998 | 0.940841 |
| DeiT-Small | 2 | 0.9167836593429668 | 0.916361 | 0.932622 | 0.934728 |
| ResNet-50 | 0 | 0.9300793051359517 | 0.932452 | 0.943019 | 0.943150 |
| ResNet-50 | 1 | 0.9291395516992046 | 0.930821 | 0.942344 | 0.941832 |
| ResNet-50 | 2 | 0.9272106712087880 | 0.929257 | 0.940724 | 0.947646 |

**Per-seed deltas** (DeiT − ResNet), pairing seed *i* against seed *i*:

| Seed | Δ macro-F1 |
|---|---|
| 0 | −0.0083934296 |
| 1 | −0.0083600484 |
| 2 | −0.0104270119 |

All three negative. ResNet-50's worst seed (0.92721) still exceeds DeiT-Small's best
(0.92169), so the families do not overlap on this metric.

---

## 5. Statistics and the pre-registered decision rule

### 5.1 The rule, as frozen

> A difference is claimed only if it is **sign-consistent across all three seeds** AND the
> pooled bootstrap 95% CI on Δ macro-F1 excludes zero. Otherwise the result is reported as
> a **tie**, interpreted through the accuracy / latency / VRAM trade-off.

Implemented in `helmet_cnn_vit.stats.decide` and unit-tested — including the case where
the interval excludes zero but the seeds disagree, which must still be a tie.

### 5.2 Pooled paired bootstrap

Crops resampled **as pairs**; within each resample each family's macro-F1 is averaged over
its three seeds, so the interval carries both crop-sampling and seed-to-seed variation.

| | |
|---|---|
| Observed pooled Δ macro-F1 | **−0.009060163301762514** |
| 95% CI | **[−0.013797334921501063, −0.004337008737569241]** |
| Confidence level | 0.95 |
| Resamples | 10,000, seeded (`seed=0`) |
| Excludes zero | **yes** (upper bound < 0) |

### 5.3 Verdict

| | |
|---|---|
| `sign_consistent` | **True** |
| `interval.excludes_zero` | **True** |
| `difference_claimed` | **True** |
| `winner` | **`resnet50`** |

Rationale, verbatim from `results.json`:

> macro-F1 differs by −0.0091 (95% CI [−0.0138, −0.0043], excludes zero) and all 3 seeds
> agree on the sign, so the difference is claimed for resnet50.

### 5.4 McNemar (exact binomial, paired, per seed)

Both models see the same crops, so their errors are paired; the exact test looks only at
the discordant crops.

| Seed | Only DeiT correct | Only ResNet correct | Both correct | Both wrong | Discordant | p |
|---|---|---|---|---|---|---|
| 0 | 204 | 246 | 6738 | 218 | 450 | **0.05315** |
| 1 | 183 | 230 | 6749 | 244 | 413 | **0.02349** |
| 2 | 210 | 270 | 6697 | 229 | 480 | **0.00702** |

All three favour ResNet-50 directionally. **Seed 0 does not clear p < 0.05** — recorded in
[§9.2](#92-seed-0-mcnemar-p--00531-does-not-clear-005). McNemar is **not** a criterion in
the frozen rule; it is reported alongside it and did not affect the verdict.

---

## 6. Calibration

Temperature fitted on **validation** and applied unchanged to test (15-bin ECE).

| Model | Seed | T | ECE before | ECE after | NLL before | NLL after |
|---|---|---|---|---|---|---|
| DeiT-Small | 0 | 1.624594 | 0.027603 | 0.007767 | 0.204013 | 0.176232 |
| DeiT-Small | 1 | 1.953502 | 0.037306 | 0.007838 | 0.231971 | 0.176324 |
| DeiT-Small | 2 | 1.789074 | 0.036853 | 0.004714 | 0.211966 | 0.173024 |
| ResNet-50 | 0 | 2.229829 | 0.035245 | 0.006516 | 0.226939 | 0.159316 |
| ResNet-50 | 1 | 2.207781 | 0.035036 | 0.006438 | 0.217439 | 0.153903 |
| ResNet-50 | 2 | 2.152413 | 0.035550 | 0.006526 | 0.210265 | 0.151509 |

Both families are **over-confident** (every T > 1), ResNet-50 more so (T ≈ 2.15–2.23 vs
1.62–1.95). Both calibrate well: ECE drops to well under 0.01 for every run. Macro-F1 is
unchanged by scaling, as expected for a monotone transform at a fixed threshold — so
calibration improves the *probabilities*, not the *decisions*.

For a system that must show a confidence to a human reviewer, this matters: raw softmax
scores from either model overstate certainty, and the fitted temperature is the correction.

---

## 7. Cost benchmark

Measured under the §12 protocol: batch 1 and 32, fp16 autocast, `torch.inference_mode()`,
100 discarded warm-up iterations, **median** of 1,000 timed iterations, CUDA events with a
synchronise at every iteration boundary, peak VRAM via `torch.cuda.max_memory_allocated()`.

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), torch 2.11.0+cu128, timm 1.0.29.

| Metric | DeiT-Small (ViT) | ResNet-50 (CNN) | Advantage |
|---|---|---|---|
| Parameters | **21,666,434** (21.67 M) | 23,512,130 (23.51 M) | DeiT, −7.9% |
| Checkpoint size | **86,714,997 B** (82.70 MiB) | 94,351,405 B (89.98 MiB) | DeiT, −8.1% |
| Batch 1 — median latency | **4.0187 ms** | 4.1538 ms | DeiT, −3.3% |
| Batch 1 — p90 latency | 8.0507 ms | 9.0633 ms | DeiT |
| Batch 1 — throughput | **248.84 img/s** | 240.75 img/s | DeiT, +3.4% |
| Batch 1 — peak VRAM | **95.979 MiB** | 110.489 MiB | DeiT, −13.1% |
| Batch 32 — median latency | **20.7662 ms** | 29.5662 ms | DeiT, −29.8% |
| Batch 32 — p90 latency | 20.9469 ms | 29.7370 ms | DeiT |
| Batch 32 — throughput | **1540.97 img/s** | 1082.32 img/s | DeiT, +42.4% |
| Batch 32 — peak VRAM | **184.487 MiB** | 280.303 MiB | DeiT, −34.2% |

DeiT-Small is cheaper on **every** measured axis. The advantage is marginal at batch 1
(~3%, where per-launch overhead dominates) and substantial at batch 32, which is the
regime a batched pipeline would actually run in.

Two measurement caveats are recorded in [§9.4](#94-benchmark-timed-random-init-models-while-checkpoint-size-came-from-trained-checkpoints)
and [§9.5](#95-batch-1-p90-is-roughly-twice-the-median).

---

## 8. Robustness

### 8.1 Per-site (test split, 12 observation sites, seed 0)

| Site | n | DeiT-Small | ResNet-50 | Δ (DeiT − ResNet) |
|---|---|---|---|---|
| Bago_highway | 188 | 0.9224 | 0.9502 | −0.0278 |
| Bago_rural | 189 | 0.8699 | 0.8685 | +0.0014 |
| Bago_urban | 336 | 0.8396 | 0.9097 | −0.0701 |
| Mandalay_1 | 2110 | 0.9027 | 0.8938 | +0.0089 |
| Mandalay_2 | 2029 | 0.9363 | 0.9342 | +0.0020 |
| Naypyitaw_1 | 168 | 0.9640 | 0.9639 | +0.0001 |
| Naypyitaw_2 | 213 | 0.8588 | 0.8962 | −0.0374 |
| NyaungU_rural | 326 | 0.8676 | 0.8841 | −0.0165 |
| NyaungU_urban | 601 | 0.9189 | 0.9316 | −0.0127 |
| Pakokku_urban | 880 | 0.8910 | 0.9229 | −0.0319 |
| Pathein_rural | 93 | 0.9235 | 0.9346 | −0.0111 |
| Yangon_II | 273 | 0.8753 | 0.8300 | **+0.0453** |

ResNet-50 leads on 8 sites, DeiT-Small on 4. Per the pre-committed interpretation rule
(§11), this **dissociation is reported per slice and not aggregated into a story**. The
largest swings in each direction — Bago_urban (−0.0701) and Yangon_II (+0.0453) — sit on
336 and 273 crops respectively and are not independently significance-tested.

### 8.2 Crop height

**Train-derived tertiles** (boundaries fixed from the training split before training):

| Bucket | n | DeiT-Small | ResNet-50 | Δ |
|---|---|---|---|---|
| < 170 px | 2475 | 0.9013 | 0.8991 | +0.0022 |
| 170–287 px | 2615 | 0.9297 | 0.9380 | −0.0083 |
| > 287 px | 2316 | 0.9358 | 0.9570 | −0.0213 |

ResNet-50's advantage grows with crop size; on the smallest tertile the families are level.

**§12's absolute buckets**, reported so the specified slice is not quietly dropped, and
**degenerate on this corpus** because HELMET boxes are whole motorcycles at 1080p:

| Bucket | n | DeiT-Small | ResNet-50 |
|---|---|---|---|
| < 32 px | 8 | 1.0000 | 1.0000 |
| 32–64 px | 7 | 0.8571 | 1.0000 |
| > 64 px | 7391 | 0.9217 | 0.9299 |

The first two buckets hold 15 crops between them and support no claim.

The tertile bucket counts differ from the pre-registration's stated table by 29 crops —
see [§9.3](#93-a-29-crop-height-bucket-mismatch-between-the-pre-registration-prose-and-the-code).

### 8.3 Synthetic corruptions

Applied to **test crops only**, at evaluation time; the same corrupted image is shown to
both families, before either model's normalisation. Nothing corrupted was ever trained on.
Seed-0 checkpoint per family.

> **Provenance note.** These numbers come from `runs/helmet_cnn_vit/report/corruptions.json`,
> **not** from `results.json`. The corruption pass is opt-in and is not wired into the
> frozen `evaluate` command, and `ExperimentReport` has no field for it. This is a
> departure from pre-registration §11's letter ("no number will appear in the write-up
> that is not in `results.json`"), disclosed here rather than resolved by silently
> omitting or silently including the slice.

| Variant | DeiT-Small | ResNet-50 | Δ |
|---|---|---|---|
| *clean (reference)* | 0.9217 | 0.9301 | −0.0084 |
| brightness s1 | 0.9180 | 0.9287 | −0.0107 |
| brightness s2 | 0.9057 | 0.9171 | −0.0114 |
| brightness s3 | 0.8672 | 0.8600 | **+0.0072** |
| gaussian_blur s1 | 0.9140 | 0.9264 | −0.0124 |
| gaussian_blur s2 | 0.8971 | 0.9011 | −0.0040 |
| gaussian_blur s3 | 0.8263 | 0.8192 | **+0.0071** |
| jpeg_compression s1 | 0.9139 | 0.9249 | −0.0110 |
| jpeg_compression s2 | 0.9036 | 0.9108 | −0.0072 |
| jpeg_compression s3 | 0.8662 | 0.8474 | **+0.0189** |
| motion_blur s1 | 0.9220 | 0.9274 | −0.0054 |
| motion_blur s2 | 0.9209 | 0.9270 | −0.0061 |
| motion_blur s3 | **not measured** | **not measured** | — |

ResNet-50 leads 8 of the 11 completed variants. DeiT-Small leads all three **severity-3**
cells — the most degraded inputs. Reported as a slice dissociation; three cells at one
severity level, untested for significance, do not support a general claim about ViT
robustness, and none is made here.

**`motion_blur` severity 3 did not run.** `robustness.py` builds a 9×9 kernel for that
severity and PIL's `ImageFilter.Kernel` accepts only 3×3 and 5×5, so it raises
`ValueError: bad kernel size`. The failure is symmetric — it removes the same cell for both
families — and was left unfixed deliberately: repairing a corruption definition after
seeing results would amend the frozen protocol. See [§9.6](#96-the-motion_blur-severity-3-cell-could-not-be-measured).

### 8.4 Day/night

**Not reported.** HELMET is daytime footage with no illumination annotation. The
`brightness` corruption is a *synthetic darkening of daylight frames*, not night data, and
is not presented as a night-time result.

---

## 9. Limitations and anomalies

These are recorded as observed. None was corrected retrospectively — repairing the
protocol after seeing the results is precisely what pre-registration exists to prevent.

### 9.1 Both selected learning rates sit on opposite grid boundaries

DeiT-Small selected **1e-4**, the grid's lower endpoint; ResNet-50 selected **1e-3**, the
upper endpoint. Neither optimum is interior to the frozen grid `{1e-4, 3e-4, 1e-3}`.

The honest reading is that **each family's true optimum may lie outside the searched
range** — DeiT-Small's below 1e-4, ResNet-50's above 1e-3 — and neither family is
demonstrated to have been tuned to its own best achievable configuration. DeiT-Small's grid
is also strongly monotone and its 1e-3 run collapsed (val macro-F1 0.652), consistent with
the ViT being the more LR-sensitive of the two under this recipe.

This does not violate the protocol: §7 fixes the grid and requires only that the budget be
**equal**, and it was — three configurations, six epochs, per family. But it does bound the
claim. The result reads as *"ResNet-50 beat DeiT-Small under this fixed, equal, three-point
LR budget"*, **not** *"a CNN beats a ViT on this task"*. The grid was **not** widened after
observing this, because widening it post hoc would break the equal-budget guarantee and
convert a pre-registered comparison into a search for a preferred answer.

### 9.2 Seed-0 McNemar p = 0.0531 does not clear 0.05

Directionally consistent with the other two seeds (both favour ResNet-50, p = 0.0235 and
p = 0.0070), but seed 0 alone is not significant at the conventional threshold.

This is disclosed because the headline claim should not be read as "significant on every
seed by every test". It changes nothing about the verdict: the frozen rule is
sign-consistency **plus** the pooled bootstrap CI, and McNemar is a reported diagnostic,
not a criterion. No post-hoc test was substituted, and the rule was not reinterpreted to
absorb this.

### 9.3 A 29-crop height-bucket mismatch between the pre-registration prose and the code

Pre-registration §9 states tertile counts of **2,475 / 2,586 / 2,345** on test. The code
produces **2,475 / 2,615 / 2,316**.

The cause is an inclusive/exclusive boundary disagreement: the prose describes the top
bucket as "287-and-over" (≥ 287), while `robustness.height_bucket` assigns the middle
bucket with `height <= high`, placing height exactly 287 px in the middle. Exactly **29
test crops have `box_h == 287.0`**, which accounts for the entire discrepancy
(2,586 − 2,615 = −29; 2,345 − 2,316 = +29).

The **code** is the executed protocol and the tables in §8.2 are what it produced; the
prose is what is inaccurate. Neither was changed. The affected slice is a secondary
robustness cut, not the primary metric, and moving 29 of 7,406 crops (0.39%) between two
adjacent buckets does not alter any conclusion.

### 9.4 Benchmark timed random-init models while checkpoint size came from trained checkpoints

`__main__.cmd_evaluate` calls `benchmark_model(spec, iters=...)` **without** a `checkpoint`
argument, so the benchmarked graphs are built with `pretrained=False` and no weights are
loaded. Consequently `benchmark.checkpoint_bytes` is `null` in `results.json`.

The checkpoint sizes reported in [§7](#7-cost-benchmark) (86,714,997 B and 94,351,405 B)
come from `ModelResult.checkpoint_bytes`, recorded at training time from the actual saved
`best.pt` files, and were verified on disk.

Weight *values* do not affect latency, throughput, or VRAM — the architecture, dtype, and
shapes do, and those are identical between the benchmarked and trained models (parameter
counts match exactly: 21,666,434 and 23,512,130). So the timing numbers stand. But the
timed artifact and the sized artifact were **not the same object**, and that is recorded
here rather than glossed. The benchmark was not modified to load checkpoints, because
changing the measurement procedure after seeing results is a protocol amendment.

### 9.5 Batch-1 p90 is roughly twice the median

8.05 ms vs 4.02 ms (DeiT) and 9.06 ms vs 4.15 ms (ResNet). Expected on a laptop GPU at
batch 1, where kernel-launch overhead, thermal behaviour, and background scheduling
dominate — which is why §12 specifies the **median**. The effect is symmetric across
families. Batch-32 p90 sits within 1% of the median, as expected once real work per
iteration dominates.

### 9.6 The `motion_blur` severity-3 cell could not be measured

`robustness.motion_blur` maps severity 3 to a 9×9 kernel; PIL's `ImageFilter.Kernel`
supports only 3×3 and 5×5 and raises `ValueError: bad kernel size`. 22 of the 24 intended
model×variant cells completed; the two missing cells are the same variant for both
families, so no asymmetry is introduced.

Deliberately **not fixed in this phase**. Changing a corruption definition after the
results are known would amend the frozen protocol; the repair belongs in a follow-up unit
that re-runs the corruption pass under an amended, re-frozen spec.

### 9.7 Training was not bitwise deterministic across repeated runs

Seeding is thorough — `derive_seed_plan` fans one base seed into decorrelated
python/numpy/torch seeds, `torch.manual_seed` and `torch.cuda.manual_seed_all` are applied,
and `cudnn.deterministic = True` is set (recorded per run in `seed_plan.json`). Despite
that, repeated training runs were **not observed to be reproducible bit-for-bit**.

The mechanism is visible in the harness: **`torch.use_deterministic_algorithms(True)` is
never called and `CUBLAS_WORKSPACE_CONFIG` is never set**, so CUDA kernels that accumulate
with non-deterministic atomics remain permitted, and fp16 autocast makes the resulting
ordering differences visible in the low-order bits. `cudnn.deterministic` constrains cuDNN
algorithm selection only; it is not full determinism.

What this **does** bound: an exact reproduction of a specific checkpoint's weights is not
guaranteed from the seed alone. What it does **not** bound: the reported comparison. The
protocol addresses run-to-run variation by design — three seeds per family, a
sign-consistency requirement, and a pooled bootstrap whose interval carries seed-to-seed
variance. Observed seed spread (sd 0.00261 and 0.00146) is smaller than the measured
family gap (0.00906), and the families do not overlap.

**Evaluation, by contrast, is exactly reproducible.** Re-running the frozen `evaluate`
against the stored predictions reproduced `results.json` and `summary.md` **byte-identically**
(sha256 `cf2e8e0a…` / `c992e467…`), confirming that every metric, statistic, calibration
value, and the verdict itself are deterministic functions of the stored run artifacts.

### 9.8 Scope limits carried from the pre-registration

- **Myanmar footage, 2016, daytime only.** No claim of transfer to Indian roads.
- **Motorcycle crops, not head crops.** The runtime classifies derived head crops; this
  experiment measures whole-motorcycle crops, which is the annotation's native
  granularity. **The winner is measured, not deployed** — see [ADR-005](adr/ADR-005.md).
- **Binary, not 3-class.** HELMET carries neither an `uncertain_occluded` nor a `turban`
  label; inventing either would fabricate supervision. Turban remains a rule-layer
  exemption in the runtime.
- **Sites are shared across the official split**, so this measures video-level rather than
  site-level generalisation. Per-site slices are reported instead of a site holdout,
  because `dataset-policy` requires preserving the authors' official split.
- **No comparison to the HELMET authors' published numbers**: the task framing differs from
  their weighted F-measure.
- **No mixup/CutMix for either family**, omitted symmetrically even though DeiT's published
  recipe includes them, because soft targets would change the loss path for one family only.
- **ConvNeXt-T / Swin-T ablation skipped** (§12 marks it optional).

---

## 10. Reproduction

```bash
export PYTHONPATH=$PWD/experiments

# 1. Corpus + official split + crops (dataset licence gate must be resolved first)
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit prepare

# 2. Tuning grid (3 LRs x 2 families x 6 epochs) + final seed runs (3 seeds x 12 epochs)
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit sweep --num-workers 6

# 3. Metrics, calibration, slices, statistics, verdict, and the cost benchmark
.venv-cnnvit/Scripts/python.exe -m helmet_cnn_vit evaluate --bench
```

The corruption pass (§8.3) has no CLI subcommand; it is invoked by calling
`helmet_cnn_vit.corrupt_eval.evaluate_corruptions` directly.

**Provenance recorded in `results.json`:** git SHA `3325e518d524b0988f53a52cba51c3ea06b6ac9c`
(tag `p4u5-prereg`), dataset `helmet-myanmar` v2020.0, torch 2.11.0+cu128, timm 1.0.29,
NVIDIA GeForce RTX 4060 Laptop GPU.

Nothing from the dataset and no model weights are committed. `data/` and `runs/` are
gitignored.

---

## 11. What may and may not be claimed

**May be claimed:**

- Under a pre-registered, leakage-safe, equal-budget protocol with a decision rule frozen
  in code before any test number existed, **ResNet-50 achieved a higher test macro-F1 than
  DeiT-Small** on HELMET driver helmet-state classification, by 0.00906 (95% pooled CI
  [−0.01380, −0.00434]), sign-consistent across three seeds.
- **DeiT-Small is the cheaper model** on the measured hardware, on every cost axis.
- Both families calibrate well after temperature scaling fitted on validation.

**May not be claimed:**

- That CNNs beat ViTs in general, or on helmet classification in general. This is one
  dataset, one geography, one three-point LR budget, two specific checkpoints.
- That either model is validated for deployment on TrafficPulse's runtime head crops. It
  is not; the crop geometry differs from what was trained.
- That the ~0.9-point gap is operationally meaningful. The protocol establishes that it is
  statistically distinguishable, not that it matters in production.
- That DeiT-Small is more robust to corruption. It leads three severity-3 cells; that is a
  dissociation worth recording, not a demonstrated property.
