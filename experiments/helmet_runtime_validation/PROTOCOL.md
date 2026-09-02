# P4-U6-V — runtime-equivalent helmet validation protocol

- **Status:** Frozen before any result was observed
- **Date:** 2026-09-01
- **Relationship to P4-U5:** **separate experiment.** P4-U5 (`p4u5-prereg`, commit `52839d0`)
  is untouched — its code, splits, checkpoints, `results.json`, and ADR-005 are read-only
  inputs here. Nothing in this document amends, reinterprets, or supersedes it.

---

## 1. The question

P4-U5 measured **whole-motorcycle crops** cut from HELMET's own annotation. TrafficPulse's
runtime classifies the **top 30% of an RT-DETR person box, at full width**. Those are
different objects. P4-U5's test macro-F1 of 0.92881 therefore says nothing about runtime
accuracy.

This experiment asks one question:

> How do the zero-shot backend, the P4-U5 DeiT-Small, and the P4-U5 ResNet-50 behave on the
> crops TrafficPulse actually produces?

It is **not** a rerun of P4-U5 and it does **not** adjudicate the CNN-vs-ViT question.

## 2. The crop pipeline (production components only)

```
frame (HELMET 1920x1080 JPEG)
  -> RTDetrDetector.detect                      [PekingU/rtdetr_r50vd, threshold 0.5]
  -> DetectionAdapter.adapt                     [serve.py LABEL_MAP, score_threshold 0.5]
  -> IouTracker.update                          [identity only; see 2.1]
  -> associate_riders                           [RiderAssociationConfig defaults]
  -> ground-truth motorcycle matching           [see 3]
  -> extract_head_region(head_fraction=0.30)    [production default]
  -> min_crop_height_px = 12.0 gate             [production default]
  -> lossless PNG on disk
```

Every component is imported from `src/trafficpulse`. Nothing is reimplemented, and no
second detection pipeline is created.

### 2.1 What tracking does and does not contribute here

The corpus samples every 5th frame, so consecutive sampled frames are 0.5s apart and are
**not** a tracking sequence. The tracker is therefore run per frame, fresh. This is
faithful for this experiment because tracking assigns *identity*; it does not alter box
geometry, and box geometry is the only thing the crop depends on. Temporal aggregation is
explicitly out of scope (§7).

## 3. Ground-truth matching rule (pre-committed)

A HELMET annotation row is a **motorcycle** box with a positional-encoding label. To attach
that label to a runtime crop, the annotated motorcycle must be matched to a detected one.

- **Match rule: IoU >= 0.50** between the annotated box and an RT-DETR `motorcycle` box,
  highest IoU wins, each detection used at most once.
- 0.50 is the standard COCO convention and is fixed **a priori**, not chosen after
  inspecting recovery rates.
- The rider crop is then the person box that `associate_riders` linked to *that* matched
  motorcycle.
- **Exactly one** associated rider is required (§4). Zero or two-plus is a recovery failure
  with its own reason code, never a silent drop.

## 4. Population

**Single-rider motorcycles only** (`rider_count == 1` in the frozen corpus).

The restriction is a consequence of an architectural fact, not a convenience:
`observations.helmet.rider_slot` returns `DRIVER` only when exactly one rider is
associated, and `UNKNOWN` otherwise, because the shipped `IouTracker` supplies no velocity
and driver-vs-pillion cannot be read off image-space position without the bike's travel
direction. P4-U5's target is the **driver** (`D`) token. On a multi-rider bike the runtime
cannot say which crop is the driver, so a driver-labelled comparison there would be
measuring a guess.

Multi-rider crops are **excluded and counted**, never discarded silently.

## 5. Protocol order (strict)

1. Derive runtime crops for **val** and **test** in one pass. Labels are attached but not
   read by any tuning step.
2. Score val with every backend. Choose each backend's operating parameter
   (`abstain_below`) on **val only**.
3. **Freeze** those parameters and record them in `frozen_operating_points.json` before the
   test split is scored.
4. Score test **once** with the frozen configuration.

No parameter is chosen, adjusted, or re-chosen after a test number is seen. No model is
retrained, and no P4-U5 hyperparameter is revisited.

### 5.1 The two operating points (both declared before any val number was seen)

**A. Forced binary choice — the primary comparison.** Every backend must commit to
`helmet` or `no_helmet` on every recovered crop. For the binary models this is the plain
arg-max. For zero-shot it is the arg-max *restricted to its `helmet` and `no_helmet`
prompts*, renormalised — a derived view of the same posterior, used because it is the only
way to compare three backends with different vocabularies on one metric. Coverage is 100%
by construction, so macro-F1 alone is comparable and no threshold is involved.

**B. Native operating point — the secondary, deployment-shaped view.** Each backend behaves
as it actually would:

- zero-shot uses its full four-prompt vocabulary; `turban` and `uncertain` are abstentions
  (PROTOCOL §8), never binary calls;
- the binary models apply an `abstain_below` floor chosen on **val**.

`abstain_below` is selected from the fixed grid `{None, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
0.80, 0.85, 0.90, 0.95}` by this rule, fixed here in advance:

> maximise **val** macro-F1 over covered crops, **subject to coverage >= 0.90**; ties break
> to the *lower* threshold (preferring coverage).

The coverage floor exists so the rule cannot buy accuracy by abstaining on everything hard,
which would be a meaningless operating point for a system that must actually decide. Both
views are reported; neither is presented alone.

## 6. Metrics

Per backend, on test: macro-F1 (primary), per-class precision/recall/F1, the confusion
matrix, coverage and abstention rate, and ECE (15 bins) where the backend emits a usable
posterior.

Paired comparisons between backends on the crops **all three** recovered: exact-binomial
McNemar, and a paired bootstrap on delta-macro-F1 (10,000 resamples, crops resampled as
pairs, seeded).

### 6.1 The P4-U5 decision rule is deliberately NOT reused

P4-U5's rule required sign-consistency **across three seeds**. That rule is inapplicable
here: this experiment evaluates **one fixed checkpoint per family** (seed 0), so there is no
seed dimension to be consistent across. Importing the rule would be a category error.

This experiment therefore makes **no adoption claim**. It reports intervals and lets the
project decide. A pre-registered adoption rule needs a seed dimension or an explicitly
justified single-checkpoint criterion, and belongs to whatever unit actually proposes
adoption.

## 7. Explicitly out of scope

- Temporal aggregation, persistence, and the no-helmet rule. This measures the **classifier
  on a crop**, not the violation decision. Reasoning stays where it is.
- Any driver/pillion heuristic.
- Any turban architecture decision.
- Retraining, threshold tuning on test, or preprocessing invented to close the domain gap
  (in particular, the JPEG q95 round-trip present in P4-U5 training is **not** reproduced —
  the goal is to measure the production path as it is).

## 8. Turban and the vocabulary asymmetry

The zero-shot backend can emit `turban`; the P4-U5 models are binary and cannot. `turban` is
**never** mapped to `no_helmet` — HELMET carries no turban label, so there is no ground truth
for it, and folding it into `no_helmet` would manufacture supervision.

Zero-shot `turban` predictions are counted and reported as a separate outcome. On the binary
ground truth they are scored as **abstentions** (no binary call made), which lowers zero-shot
coverage rather than silently converting a religious head covering into a violation. The
resulting comparison is therefore **not** a like-for-like vocabulary match, and §12 of the
report quantifies that.

## 9. Denominator honesty

Two numbers are reported and never conflated:

- **Conditional accuracy** — performance on crops the pipeline successfully recovered. This
  is what the classifier does when it is given something.
- **End-to-end coverage** — the fraction of eligible annotated riders that produced a crop
  at all.

A high conditional accuracy on a low coverage is not a working system, and the report says
so explicitly rather than leading with the flattering number.
