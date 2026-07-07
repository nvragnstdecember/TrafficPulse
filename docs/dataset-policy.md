# TrafficPulse Dataset Split & Leakage Policy

**Version:** 1.0.0 · **Status:** frozen (Phase 0-F, U4)
**Companion:** [`docs/evaluation-protocol.md`](evaluation-protocol.md) · **Registry:** [`registry/`](../registry/)
**Canonical architecture reference:** `docs/architecture-review.md` (§8, §16, §24)

This document freezes the data-governance and leakage-prevention policy **before**
any dataset is downloaded, prepared, split, or used for training. It defines
policy only — U4 implements no split-generation code. Split generation, frame
extraction, and manifests are Phase 1 work executed **under** this policy.

## 1. Purpose and status
Prevent train/test leakage and licence violations, and make every split
reproducible and auditable. Frozen for Phase 0-F; changes require review (§15).

## 2. Core leakage principle
A model or rule must never be evaluated on data correlated with its training
data. Adjacent video frames are highly correlated, so the unit of splitting is a
**source group**, never an individual frame. Test data measures generalization,
not memorization.

## 3. Group-based split requirement (FROZEN)
- **A.** Never randomly split extracted frames from the same video across
  train/validation/test.
- **B.** Group by the highest meaningful source unit available: **camera →
  video → sequence → route/session → scene**, depending on dataset structure.
- All frames/crops/tracks derived from one source group stay entirely within one
  split.

## 4. Split unit hierarchy
Choose the coarsest defensible unit the dataset supports, in priority order:
1. **site / camera** (whole-site holdout — required for generalization claims);
2. **video / recording session**;
3. **sequence**;
4. **image** (only for genuinely independent stills, e.g. CCPD).
Derived crops **inherit** their source video's split; **no track spans splits**.

## 5. Camera / video / sequence grouping
Every registry entry records `split_leakage.source_grouping_unit` and the
availability of camera / video / sequence identity. The grouping unit actually
used for a split must be recorded in the split manifest (§13). Where camera
identity is unknown, fall back to the coarsest known unit and record the
limitation.

## 6. Frame extraction policy
Frame extraction is a Phase 1 step performed **after** the split is decided at
the source-group level. Frames are extracted within an already-assigned group;
extraction never re-mixes frames across groups. Extraction parameters (stride,
resolution) are recorded with the split manifest.

## 7. Near-duplicate frame risk
Temporally adjacent frames and near-duplicate captures are treated as the same
group. **H.** Near-adjacent frames from the same temporal sequence must not leak
across splits. Where near-duplicates may exist across nominally independent
stills, apply a grouping/dedup check and record it.

## 8. Track / vehicle identity leakage
A single tracked entity (vehicle/rider) must not appear in more than one split.
No track spans splits; crops derived from a track inherit that track's split.
Where a subject/vehicle re-appears across videos, prefer the coarser (site)
grouping to avoid identity leakage.

## 9. Official dataset split handling
- **C.** Preserve official dataset splits where they are authoritative and
  appropriate, unless a documented experimental reason requires otherwise.
- When an official split is used, record that fact in the manifest; when it is
  overridden, record the reason. `preserve_official_split` in each registry
  entry states the intended handling.

## 10. Cross-dataset evaluation
Pretraining on one dataset (e.g. CCPD detection, HELMET helmet) and evaluating
on another (own footage) is encouraged for domain-shift measurement, but the
**evaluation** dataset's holdout discipline still governs; a component may not be
tuned on any footage it will later be evaluated on.

## 11. Domain-shift evaluation
Generalization claims (e.g. "works on Indian fixed-camera roads") require a
**whole-site holdout** in the evaluation footage — a site never seen in training.
Domain gaps recorded per entry (`task_fit.domain_gaps`) frame what each dataset
can and cannot validate.

## 12. Annotation revision policy
Annotations are versioned with the dataset (§15). A re-annotation that changes
labels or semantics is a dataset version change and triggers review of dependent
splits, models, and evaluations. Ontology semantics come from U3 and are not
changed to fit a dataset.

## 13. Split manifest requirements (FROZEN)
- **D.** Custom splits are generated from a **versioned split manifest**, never
  by ad-hoc directory moves.
- **E.** The manifest preserves **source identifiers** (site/camera/video/
  sequence/track ids) for every item and is **reproducible** from a seed.
- A manifest records: dataset id + version, grouping unit, per-group split
  assignment, seed, generation command, and the tool/version that produced it.

## 14. Hashing / provenance requirements
No dataset is downloaded in U4 (`local_acquisition_status: not_downloaded` for
all entries). At authorized acquisition time, record source checksums and an
immutable source reference; the split manifest and any derived artifacts are
content-addressed so a result can be traced to exact inputs.

## 15. Dataset versioning
Each dataset carries a version/release identifier. Any change to data or
annotations increments the version and triggers review of affected splits,
models, and evaluations. The registry entry is updated, never silently edited to
misrepresent history.

## 16. Test-set protection (FROZEN)
- **F.** Test data must not be used for model selection, threshold tuning,
  hyperparameter tuning, or repeated informal debugging.
- The test split is opened only for final reporting. Repeated peeking is
  leakage.

## 17. Hyperparameter-tuning boundary
- **G/H reinforcement:** all tuning — learning rates, augmentation, and **rule
  thresholds** (temporal windows, grace periods, dwell limits) — is done on the
  **validation** split only. Tuning any threshold on test footage is leakage.

## 18. Evaluation-footage separation
- **G.** Event-level evaluation footage is kept **isolated** from
  component-model development. The `event-evaluation-footage` entry is held out
  from component training; demo footage is separate from held-out evaluation
  footage.

## 19. Custom footage handling
Own footage requires institutional permission/ethics clearance before recording,
a written data-handling plan, restricted/logged storage of originals, redaction
of incidental PII, a retention policy, and a deletion checklist. No permission or
site is claimed to exist unless the repository records it (currently none).

## 20. Reproducibility requirements
Splits are deterministic from a seeded, committed manifest; evaluation commands
are repeatable; dataset checksums and manifest hashes tie results to exact
inputs. **I.** The CNN-vs-ViT helmet comparison uses **identical split
assignments** and equivalent evaluation conditions across both model families.
**J.** The source-grouping metadata required by U3 is preserved end-to-end.

---

### Frozen principle checklist (A–J)
- **A** No random frame-level splitting across videos — §3.
- **B** Group by highest meaningful source unit — §3–§4.
- **C** Preserve authoritative official splits (documented overrides) — §9.
- **D** Custom splits from a versioned manifest — §13.
- **E** Manifests preserve source ids and are reproducible — §13.
- **F** Test set not used for selection/tuning/debugging — §16.
- **G** Event-evaluation footage isolated from component development — §18.
- **H** No near-adjacent-frame leakage across splits — §7.
- **I** CNN-vs-ViT uses identical splits and equivalent conditions — §20.
- **J** U3 source-grouping metadata preserved — §20.
