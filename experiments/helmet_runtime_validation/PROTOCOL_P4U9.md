# P4-U9 — runtime classifier re-evaluation on the corrected recovery population

- **Status:** Frozen 2026-09-01, after P4-U8's recovery counts were known and **before any
  classifier was run on the corrected crops**.
- **Relationship to prior units:** P4-U5 (`p4u5-prereg`, commit `52839d0`), P4-U6-V, P4-U7
  and P4-U8 are **read-only inputs**. None is rerun, amended, or reinterpreted. This unit
  writes only to `runs/helmet_runtime_validation_p4u9/`.

---

## 1. The question

P4-U8 corrected the evaluation scaffold and the recovered population changed materially:

| split | P4-U6-V recovered | P4-U8 recovered | change |
|---|---|---|---|
| val | 1,236 / 2,284 (54.1%) | 1,704 / 2,284 (74.6%) | +468 |
| test | 2,292 / 4,269 (53.7%) | 3,175 / 4,269 (74.4%) | +883 |

The crops that entered are not a random sample of the old ones: they are systematically
smaller (median head region 59.7px -> 53.5px on test) and less class-skewed
(`no_helmet` share of the recovered set 22.4% -> 24.0% against an eligible share of 26.3%).
A macro-F1 measured on the old population therefore does not carry over. This unit asks:

> After correcting the evaluation scaffold, how do the zero-shot backend, the P4-U5
> DeiT-Small and the P4-U5 ResNet-50 perform on the crops TrafficPulse actually produces?

PROTOCOL.md section 5 (P4-U6-V) permitted re-scoring only if the corrected population is
materially different. The table above is the pre-declared justification, and it was
established without any classifier being run.

## 2. What is held fixed

Nothing about any model changes. Specifically unchanged and re-used byte-for-byte:

- the three backends and their checkpoints — `openai/clip-vit-base-patch32` (zero-shot,
  production seam), `runs/helmet_cnn_vit/final/resnet50_lr0.001_s0/checkpoints/best.pt`,
  `runs/helmet_cnn_vit/final/deit_small_lr0.0001_s0/checkpoints/best.pt` (both P4-U5
  seed 0);
- the scoring code (`score.py`), including preprocessing, so a crop reaches each model
  exactly as it did in P4-U6-V;
- the detector, its 0.50 threshold, the tracker, the association policy, the 0.30 head
  fraction and the 12px minimum-height gate;
- the frozen P4-U5 val/test split.

No model is retrained. No detector threshold is touched. No preprocessing is invented.

## 3. Population

The **P4-U8 rider-inclusive recovered population**: single-rider motorcycles whose
annotation matched a runtime evaluation proxy at IoU >= 0.50 and which carried exactly one
associated rider and cleared the production head-crop gates. Val 1,704 crops; test 3,175.

Crops are re-cut from the original HELMET frames with the production
`extract_head_region`, from the **same** rider boxes P4-U8 recorded. The detector is not
re-run: the boxes come from P4-U7's detection record, whose fidelity P4-U8 proved by
reproducing P4-U6-V's recovered crop-id set exactly.

**All three backends see the same PNG files**, decoded identically, so any difference
between them is a property of the model and not of the input.

Multi-rider motorcycles remain out of scope and are reported separately, never absorbed
(section 8).

## 4. Protocol order (strict)

1. Re-cut the corrected crops for val and test in one pass. Labels travel with the crops
   but are read by no tuning step.
2. Score **val** with every backend.
3. Choose each backend's `abstain_below` on **val only**, by the rule in section 5.2, and
   write it to `frozen_operating_points.json`.
4. Score **test** once, with the frozen configuration.

No parameter is chosen, adjusted, or re-chosen after a test number is seen.

## 5. The two operating points

Both are inherited verbatim from P4-U6-V section 5.1 — reusing an already pre-registered
rule rather than inventing a new one — and both are declared here before any val number on
the corrected population was seen.

**A. Forced binary choice — the primary comparison.** Every backend commits to `helmet` or
`no_helmet` on every recovered crop. For the binary models this is the plain arg-max; for
zero-shot it is the arg-max restricted to its `helmet` and `no_helmet` prompts and
renormalised — a derived view of the same posterior, and the only way to put three
different vocabularies on one metric. Coverage is 100% by construction, so macro-F1 alone
is comparable and no threshold is involved.

**B. Native operating point — the secondary, deployment-shaped view.** Zero-shot uses its
full four-prompt vocabulary and treats `turban` and `uncertain` as abstentions; the binary
models apply an `abstain_below` floor.

### 5.2 Threshold selection (val only)

`abstain_below` is selected from the fixed grid `{None, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
0.80, 0.85, 0.90, 0.95}` by the rule:

> maximise **val** macro-F1 over covered crops, **subject to coverage >= 0.90**; ties break
> to the *lower* threshold (preferring coverage).

The coverage floor stops the rule buying accuracy by abstaining on everything hard.

### 5.3 A pre-declared sensitivity check

P4-U6-V's frozen thresholds (zero-shot `None`, ResNet 0.80, DeiT 0.60) are **also** applied
in the same single test pass, and reported whatever they show. Declaring this now, before
test is read, is what keeps it a sensitivity check rather than a second look at test. The
val-selected values remain the primary result; the inherited values are reported beside
them so a reader can see whether the conclusion depends on the selection.

## 6. Metrics

Per backend, on test: macro-F1 (primary), accuracy, per-class precision / recall / F1, the
confusion matrix, coverage, abstention rate, and ECE (15 bins) where the backend emits a
usable posterior. Metric code is **imported** from the frozen P4-U5 package; nothing is
reimplemented.

Paired comparisons on the crops all three backends recovered (which, since all three score
the same file set, is the whole recovered population):

- exact-binomial **McNemar**;
- a **paired bootstrap** on delta-macro-F1, 10,000 resamples, crops resampled as pairs,
  seed 0, using the frozen P4-U5 implementation.

### 6.1 The McNemar implementation

The frozen `helmet_cnn_vit.stats.mcnemar` evaluates `sum(comb(n, i)) / 2.0**n` in float and
raises `OverflowError` once the discordant count exceeds about 1,023. P4-U5 never hit that;
this comparison does, because zero-shot and the trained models disagree on thousands of
crops. The frozen helper is **not patched** — it belongs to a tagged, published experiment.
The validated log-space implementation already written for P4-U6-V
(`analyse.exact_mcnemar`, asserted equal to the frozen one wherever that one can run) is
used instead, and this paragraph is the record of why.

### 6.2 Comparability with P4-U6-V

P4-U6-V's numbers were measured on a different (smaller) crop population and are **not**
directly comparable. Two things are therefore reported:

- the corrected numbers on the full corrected population (the headline);
- the same numbers restricted to the **shared subset** — crops recovered under *both*
  conventions — which is the only like-for-like comparison available.

Any statement that a backend "improved" or "degraded" relative to P4-U6-V is made on the
shared subset or not at all.

### 6.3 No adoption rule is imported

P4-U5's decision rule required sign-consistency across three seeds. This experiment
evaluates **one fixed checkpoint per family** (seed 0), so there is no seed dimension and
importing the rule would be a category error. This unit therefore makes **no adoption
claim**: it reports effect sizes, intervals and the qualitative factors of section 7, and
the adoption decision is made separately and explicitly.

A statistically significant metric difference is **not** by itself a production winner.

## 7. What the adoption discussion must cover

Reported alongside the metrics, because a macro-F1 ranking does not settle deployment:

- **effect size** — the magnitude of delta-macro-F1, not only its significance;
- **calibration** — ECE, and whether a posterior can be trusted by the abstention gate;
- **abstention** — how much each backend declines to answer, and on which crops;
- **vocabulary compatibility** — the turban requirement of section 9;
- **deployment compatibility** — whether a production wrapper exists at all;
- **statistical uncertainty** — bootstrap intervals and McNemar p-values, read together;
- **the one-checkpoint limitation** — a single seed per family cannot separate an
  architecture effect from a checkpoint effect.

## 8. Multi-rider (unchanged, and not papered over)

`observations.helmet.rider_slot` returns `DRIVER` only for exactly one associated rider and
`UNKNOWN` otherwise. No driver/pillion attribution is invented here. Single-rider crops
remain the controlled evaluation population; the multi-rider share of the frozen corpus and
the rider-count distribution are reported, and no result on single-rider crops is presented
as evidence about multi-rider traffic.

## 9. Turban (unchanged, and quantified)

`turban` is **never** mapped to `no_helmet` and the capability guard is never bypassed.
HELMET carries no turban label, so there is no ground truth for it; folding it into
`no_helmet` would manufacture supervision and would systematically produce false violations
against a religious group.

Zero-shot `turban` predictions are counted and reported as their own outcome, and on the
binary ground truth they are scored as **abstentions**, which lowers zero-shot coverage
rather than converting a head covering into a violation. This unit reports the number and
percentage of turban predictions on the corrected population, their effect on zero-shot
coverage, and what the binary models can and cannot replace. It does **not** choose a
turban architecture.

## 10. Explicitly out of scope

Temporal aggregation, persistence and the no-helmet violation rule (this measures the
classifier on a crop, not the violation decision); any driver/pillion heuristic; any turban
architecture decision; retraining; threshold tuning on test; and any preprocessing invented
to close the domain gap — in particular the JPEG q95 round-trip present in P4-U5 training
is still **not** reproduced, so that residual domain gap remains and is stated in the
report.
