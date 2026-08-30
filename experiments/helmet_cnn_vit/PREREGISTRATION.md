# Pre-registration — mandatory CNN-vs-ViT helmet experiment (P4-U5)

- **Status:** Frozen
- **Date:** 2026-08-29
- **Baseline commit:** `5cad176`
- **Authority:** `TRAFFICPULSE_MASTER_SPEC.md` §4; `docs/architecture-review.md` §12;
  `docs/evaluation-protocol.md` §8; `docs/dataset-policy.md` principle I; `docs/phase-4-plan.md` P4-U5

Everything in this document is fixed **before the first training run**. It is committed and
git-tagged so that nothing below can be claimed to have been chosen after seeing a result.
Deviations from architecture-review §12 are listed in full in the last section rather than
discovered at write-up time.

---

## 1. Task

Binary classification of **driver helmet state** — `helmet` vs `no_helmet` — from the crop of one
annotated motorcycle.

HELMET annotates one box per tracked motorcycle, labelled with a positional-encoding string that
names every rider position and its helmet state (`DHelmet`, `DNoHelmetP1NoHelmet`, and 34 more).
The target is the `D` token, parsed by `helmet_cnn_vit.labels.parse_label`. Passenger states are
**not** part of the target; they are recorded per crop (`rider_count`, `any_no_helmet`) as
covariates only.

The task is defined at the annotation's native granularity deliberately. Deriving head boxes to
match the runtime's head crops would inject an unvalidated geometric heuristic into the label
pathway, and with pillion passengers the derived region often contains the wrong head while the
label is the driver's. That would confound the comparison.

## 2. Data

| | |
|---|---|
| Dataset | HELMET (Siebert & Lin), OSF node `4pwj8` |
| Licence | **CC-BY-4.0**, verified 2026-08-29 from the OSF API (licence object `563c1cf88c5e4a3877f9e96a`) |
| Attribution (required) | Siebert, F.W. and Lin, H. — HELMET dataset (OSF, https://osf.io/4pwj8/), CC-BY 4.0 |
| Scale | 910 clips, 12 observation sites, 100 frames/clip @ 10 fps, 1920x1080 |
| Annotations | 283,377 rows, 10,006 tracks, 36 label strings; `annotation.zip` sha256 `6b7f2245...af9e55c` |

Nothing from the dataset is committed to this repository. `data/` is gitignored.

## 3. Sampling policy (frozen)

| Parameter | Value |
|---|---|
| `frame_stride` / `frame_offset` | 5 / 1, giving frames 1, 6, ..., 96 (20 of 100 per clip) |
| `max_crops_per_track` | 6, evenly spaced across the track's surviving frames |
| `min_box_side_px` | 16 (shorter side) |

Rationale: a track is annotated on up to 100 consecutive frames at 10 fps, so its crops are
near-duplicates. Sampling stops a handful of long tracks from dominating the loss and stops the
corpus from looking seven times larger than it is. Every excluded row is **counted** in
`CorpusStatistics`, never silently dropped.

**Realised corpus:** 39,965 crops from 10,006 tracks (215 tracks excluded by the size floor).

- `corpus_hash` = `ad42119c67be87dd5ec0203096d7a6256380ddf7e1e3ba99d82aa7d75bc11147`

## 4. Splits

The authors' **official** `data_split.csv` (sha256 `d9cdaa6a...8fcfe8f7`) is applied verbatim.
`dataset-policy` requires preserving an official split, and it is video-level, which satisfies
§12's "split unit = source video".

| Split | Videos | Crops | helmet | no_helmet | no_helmet share |
|---|---|---|---|---|---|
| train | 636 | 28,470 | 20,165 | 8,305 | 29.17% |
| val | 92 | 4,089 | 2,941 | 1,148 | 28.08% |
| test | 182 | 7,406 | 5,315 | 2,091 | 28.23% |

- Split-manifest sha256 = `ce53d29b8a53d7be007a485e023e9d3875a882db6b0cd486f565dbe5c558b626`
- Independently re-validated by H3's `SplitValidator` with `VideoAwareGrouping`: no video, no
  track, and no frame straddles a split.
- **Tuning and thresholds use the validation split only.** The test split is read exactly once per
  run, after training completes.

## 5. Models

| | CNN | ViT |
|---|---|---|
| Name | ResNet-50 | DeiT-Small |
| Checkpoint | `timm/resnet50.a1_in1k` | `timm/deit_small_patch16_224.fb_in1k` |
| Init | ImageNet-1k | ImageNet-1k |
| Licence | Apache-2.0 | Apache-2.0 |
| Input | 224px square-padded crop | 224px square-padded crop |

Both are built by the same `create_model` call and trained by the same loop. Normalisation is the
one deliberate per-family difference: each checkpoint is fed the mean/std it was pretrained with,
resolved from its own `pretrained_cfg`. Forcing a shared normalisation would feed at least one
model inputs it was never trained for.

## 6. Preprocessing and augmentation

Crops are cut tight to the annotated box (no context expansion), clamped to the frame, zero-padded
to a square, resized to 224px, and stored as JPEG q95. Identical for both families, applied before
any split is consulted.

| Family | Recipe |
|---|---|
| Both | RandomResizedCrop scale (0.7, 1.0); horizontal flip p=0.5; colour jitter 0.2 |
| ViT only | plus RandAugment `rand-m7-n2-mstd0.5` (DeiT's published recipe) |

Evaluation transforms have **no randomness**. Per evaluation-protocol §8, identical recipes are
not required and would be unfair; what must be equal is the tuning budget.

## 7. Training budget (equal across families)

| | |
|---|---|
| Tuning | LR in {1e-4, 3e-4, 1e-3}: **3 configs per family**, 6 epochs each |
| Selection | highest **validation** macro-F1, never test |
| Final | selected config retrained for 12 epochs at **seeds 0, 1, 2** |
| Optimiser | AdamW, weight decay 0.05, 1 warm-up epoch then cosine decay to 1% of peak |
| Batch / precision | 64, fp16 AMP |
| Imbalance | class-weighted cross-entropy from **train** counts; no test-set rebalancing |

## 8. Metrics

**Primary: macro-F1.** Also: balanced accuracy, accuracy, per-class precision/recall/F1,
PR-AUC(`no_helmet`), the 2x2 confusion matrix, ECE (15 bins) with reliability diagram, and
temperature scaling fitted on validation and reported pre/post.

Cost: parameter count, checkpoint bytes, latency at batch 1 and 32 (fp16, `inference_mode`, 100
warm-up iterations, median of 1,000 timed iterations), throughput, and peak VRAM via
`torch.cuda.max_memory_allocated`.

A class absent from a slice's ground truth reports `None` and is excluded from macro averages; a
class present but never predicted correctly honestly scores `0.0` (H5's convention).

## 9. Robustness slices

- **Corruptions:** Gaussian blur, motion blur, JPEG compression, brightness, at three severities each.
- **Per-site:** all 12 observation sites, on the test split.
- **Crop height:** reported two ways.
  - §12's absolute buckets (under 32 / 32-64 / over 64 px). On this corpus these are
    **degenerate** — 8 / 7 / 7,391 on test — because HELMET boxes are whole motorcycles at 1080p,
    not head crops. Reported anyway, so the specified slice is not quietly dropped.
  - Tertiles derived from the **training** split only, before training: under 170 / 170-287 /
    287-and-over px, giving 2,475 / 2,586 / 2,345 on test. This is the slice that can actually
    support a claim.
- **Day/night:** not reported. HELMET is daytime footage; the counts do not permit the slice.

## 10. Statistics and the decision rule (pre-committed)

Both models are evaluated on the same test crops, so their errors are paired.

- **McNemar** (exact binomial) per seed, over the discordant crops.
- **Paired bootstrap** on delta-macroF1: 10,000 resamples, crops resampled as pairs, seeded.
- **A difference is claimed only if it is sign-consistent across all three seeds AND the pooled
  bootstrap 95% CI on delta-macroF1 excludes zero.** Otherwise the result is reported as a
  **tie**, interpreted through the accuracy / latency / VRAM tradeoff.

This rule is implemented in `helmet_cnn_vit.stats.decide` and unit-tested, including the case
where the interval excludes zero but the seeds disagree, which must still be a tie.

## 11. Pre-committed interpretation

- A tie is a legitimate outcome and will be reported as one. It is **not** evidence that the two
  architectures are equivalent.
- Slice dissociations (a family better on one site or corruption, worse on another) are reported
  per slice and not aggregated into a story.
- Interpretation refers to the data-scale literature on CNN-vs-ViT sample efficiency, not to
  post-hoc rationalisation of whatever we observe.
- No number will appear in the write-up that is not in `results.json`.

## 12. Deviations from architecture-review §12

| §12 requires | This experiment does | Why |
|---|---|---|
| 3-class (helmet / no_helmet / uncertain_occluded), turban a 4th at 150+/split | **Binary** helmet / no_helmet | HELMET carries neither an `uncertain` nor a `turban` label. Inventing either would fabricate supervision. Turban remains a rule-layer exemption in the runtime. |
| AI City T5 + HELMET + custom Indian crops | **HELMET only** | AI City is recorded PROPRIETARY / REJECTED; custom Indian footage has no ethics or permission clearance on file. |
| Whole-site holdout test set | Official split plus **per-site test slices** | The official split is video-level and shares all 12 sites across splits. Preserving it is required by dataset-policy; per-site slices measure site generalisation without discarding it. |
| Up to 8 val-selected configs per family | **3 per family** | Time budget. The fairness constraint is that the budget is *equal*, and it is. |
| Head crops | **Motorcycle crops** | See section 1. |
| Each family's best-known recipe | Shared base + RandAugment for the ViT; **no mixup/CutMix for either** | Mixup needs soft targets, which would change the loss and metric path for one family only. Omitted symmetrically. |
| ConvNeXt-T / Swin-T ablation pair | **Skipped** | §12 marks it optional. |
| Native crop height as a covariate | Recorded per crop, and used for the height slices | No deviation; noted for completeness. |

## 13. Disclosures

- Before the corpus was complete, a short smoke run of both families was executed on the 133 clips
  in `part_1.zip` **to validate the harness on GPU**: that the loop runs, AMP works, and
  checkpoints and predictions are written. No hyperparameter, model, or analysis choice was taken
  from it; the tuning grid in section 7 was fixed a priori.
- Hardware: RTX 4060 Laptop (8 GB), i7-13620H, 16 GB RAM, Windows 11. torch 2.11.0+cu128, timm
  1.0.29, in an isolated `.venv-cnnvit` that does not affect the application environment.

### When this document was frozen

This protocol was committed and git-tagged **before the final 3-seed runs**, not before all
training. At the moment of freezing, **three validation-only tuning runs of DeiT-Small had
already been observed** (the learning-rate grid of section 7, scored on validation only).

One correction was made after those observations: `stats.py` was changed to implement the
**pooled** bootstrap that section 10 and architecture-review §12 specify, replacing an earlier
implementation that bootstrapped a single seed. The single-seed form would have described one
run rather than the family and would have understated the uncertainty a claim must survive; the
correction brings the code into line with the protocol as written, and makes the interval more
conservative, not less.

**Unchanged by that correction, and unchanged since this document was first written:** the claim
rule (`stats.decide`), the model pair, the learning-rate grid, the seeds, the split, the primary
metric, and the selection criteria.

**Not known at the time of the correction:** any ResNet-50 result (that family had not been
trained at all), any final-seed result, any test-split evaluation, any model selection, and any
confidence interval. No number from the held-out test split existed.
