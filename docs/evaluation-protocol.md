# TrafficPulse Evaluation Protocol

**Version:** 1.0.0 · **Status:** frozen (Phase 0-F, U4)
**Companion:** [`docs/dataset-policy.md`](dataset-policy.md) · **Architecture ref:** `docs/architecture-review.md` (§12, §17, §22, §23)

This is a **protocol specification**, not evaluation code. It defines *what* is
measured and *how comparisons are made valid*, so that later implementation
(Phase 1+) has a frozen target. U4 implements no metrics, no matching code, and
no TrackEval/OCR/detector integration. No benchmark numbers are asserted here.

## 1. Purpose and status
Freeze the evaluation design before any results exist, so metrics, splits, and
comparison rules cannot be chosen post-hoc to flatter a result. Frozen for
Phase 0-F; changes require review.

## 2. Evaluation layers
Layers are evaluated **separately**; there is no single "overall accuracy":
detection · tracking · helmet classification (incl. CNN-vs-ViT) · signal-state
classification · ANPR (detection + OCR) · speed feasibility · event-level
violations · system performance · human review · robustness slices.

## 3. Dataset / split prerequisites
All evaluation obeys [`dataset-policy.md`](dataset-policy.md): group-based
splits, whole-site holdout for generalization, test-set protection, validation-
only tuning (including rule thresholds). Event-level metrics require the held-out
own footage (`registry/datasets/event-evaluation-footage.yaml`), which does not
yet exist.

## 4. Reproducibility requirements
Every reported result records: dataset id + version, split manifest hash, model
name + version + weights hash, code version (git SHA), config, seed(s), and the
exact command. Results are reproducible from these. Rule replay is deterministic
from the observation log (architecture §15).

## 5. Detection evaluation
Protocol-level metrics: precision, recall, **AP / mAP** (COCO-style mAP@[.5:.95]
and AP50), **per-class AP**, and **size-stratified AP** (small/medium/large — the
small-object case matters most). Report the IoU convention used for matching and
the confidence/operating threshold at which P/R are quoted. Report per
**domain slice** (site/viewpoint/lighting) where metadata permits. No metric code
in U4; no invented numbers.

## 6. Tracking evaluation
Protocol-level identity/association metrics: **HOTA**, **IDF1**, **MOTA**, and
**ID-switch count**, with ID switches counted **inside violation-relevant zones**
(architecture §23). HOTA balances detection and association; IDF1 emphasizes
identity consistency — both are reported, not one alone. Intended reference tool
is TrackEval; **no TrackEval integration exists** and none is built in U4.

## 7. Helmet-classification evaluation
Per-class **precision/recall/F1**, **macro-F1** (primary), **balanced accuracy**,
and a **confusion matrix** over the U3 labels. `turban` is reported as a
**distinct** visual state; `uncertain` is an **abstention** with reported
**coverage** (fraction abstained) and **selective accuracy** on the covered set.
**`uncertain` is never collapsed into `no_helmet`; `turban` is never collapsed
into `no_helmet`.** Robustness slices: blur, occlusion, small crops (by head-crop
pixel height), illumination, and truncation where labels support it.

## 8. CNN-vs-ViT comparison protocol (mandatory experiment)
The architecture's mandatory experiment (helmet-state classification) requires,
for both families:
- same task; same U3 ontology; **same train/validation/test split assignments**;
- **same source grouping** (crops inherit source-video split; no track spans
  splits);
- **equivalent augmentation policy** (shared base + each family's best-known
  recipe under an equal tuning budget — identical recipes are not required and
  can be unfair);
- comparable input-resolution policy (or an explicitly justified difference);
- documented **pretrained initialization**, **parameter count**, **training
  compute**, and **training duration**;
- **same primary metrics** (macro-F1) and **same robustness slices**;
- **inference latency**, **throughput**, and **memory/VRAM** reporting;
- **statistical uncertainty**: multiple seeds; a difference is claimed only if it
  is sign-consistent across seeds and a bootstrap CI on ΔmacroF1 excludes zero —
  otherwise reported as a tie interpreted through accuracy/latency/VRAM tradeoffs.
The winner is **not** selected now and nothing is trained in U4.

## 9. Traffic-signal classification evaluation
Per-class P/R/F1 and confusion matrix over the U3 signal states
(`red/amber/green/off/unknown`); `unknown` is an **abstention** with reported
coverage. Report calibration/confidence gating behavior and performance at
state-transition boundaries. A heuristic (HSV) baseline is compared against any
learned classifier. Night is a robustness slice only (§23).

## 10. ANPR evaluation (two separate stages)
**(1) Plate detection:** precision/recall/AP where labels permit, sliced by plate
pixel-width and blur. **(2) OCR recognition:** **exact full-string match**,
**character error rate (CER)** and character accuracy, **format-valid rate**, and
**single-frame vs multi-frame consensus** comparison. Slices: plate
size/visibility/blur, and script/layout (esp. two-line motorcycle plates) where
data supports it; failure analysis by slice. **Privacy:** plate strings are
personal data — evaluated only on authorized footage; non-target plates redacted;
covered by retention. No Indian ANPR dataset is claimed sufficient — real Indian
transcriptions for evaluation must be obtained (registry: synthetic for training,
own footage for eval).

## 11. Speed-estimation feasibility evaluation
Controlled calibrated-zone experiment only (architecture §17); general monocular
speed is **not** claimed. Design: a **calibrated scene**; **trustworthy ground
truth** (e.g. GNSS-logged passes, stopwatch cross-check); **representative
trajectories**; **perspective** and **distance** variation; **repeated passes**
where practical. Metrics: **MAE**, **median absolute error**, **P95 absolute
error**, **signed bias**, per speed band; **calibration-quality reporting**
(reprojection RMSE) and documented **failure conditions**. Every reported speed
is `v ± kσ`.

### Provisional candidate targets — NOT accepted requirements
The architecture's candidate targets **MAE ≤ 3 km/h** and **P95 absolute error ≤
6 km/h** (20–60 km/h band, ≥ ~20 passes) are recorded here as **provisional
candidate engineering targets for the controlled calibration experiment only**.
They are **not** accepted project requirements. **Evidence required before formal
acceptance:** an empirical error distribution from the calibrated-scene
experiment on held-out passes at the demo site, with the band, pass count, and
uncertainty method justified. If the gate fails, speeding is demoted to an
honest experiment chapter and excluded from penalty simulation. The gate is
**not** claimed passed.

## 12. Event-level violation evaluation
Evaluated per violation type: `no_helmet`, `triple_riding`, `red_light_jumping`,
`wrong_way`, `illegal_stopping`, `speeding`. A predicted event matches a
ground-truth event iff **(same violation type)** AND **(temporal overlap)** AND
**(track/entity compatibility where ground truth supports it)**. Requires the
held-out own footage with event-level temporal ground truth.

## 13. Event matching semantics
- Matching predicate: same violation type; temporal overlap by **temporal IoU**
  (or a minimum temporal-intersection fallback); track/entity compatibility
  (e.g. median track-IoU over the overlap, or a center-containment fallback)
  where GT supports it.
- **Temporal IoU ≥ 0.3 is a *provisional candidate default*, not a universal
  constant.** It must be validated against annotation granularity and typical
  event duration before acceptance; short events may need a minimum-intersection
  rule instead. The chosen value and its justification are recorded per report.
- **One-to-one** matching: greedy by descending prediction confidence.
- **Duplicate** predictions matching an already-matched GT event count as false
  positives (a duplicate-rate is reported).
- **Unmatched predictions → false positives; unmatched GT events → false
  negatives.**
- Abstentions are logged and counted, never silently dropped.

## 14. False events per hour
Report **false events per hour** per violation type as a headline operational
metric (a rate the deployment cares about), alongside precision/recall/F1.

## 15. Clean-footage evaluation
Run on footage containing **no** violations of a given type to measure the
false-event rate directly; a rule that fires on clean footage is penalized
regardless of its recall elsewhere.

## 16. Per-violation reporting
Report **per-violation** event precision/recall/F1, false-events/hour, median
detection delay (t_confirm − t_start_GT), duplicate rate, and evidence
completeness, plus a **macro** summary across violations. No single blended score.

## 17. Confidence calibration evaluation
Where a score is used as a probability, report **ECE** and reliability diagrams,
and temperature scaling fit on validation (reported pre/post). A confidence
score is **not** presented as a calibrated probability unless calibration is
demonstrated (architecture §10, §13).

## 18. Ablation expectations
Report justified ablations, e.g. single-frame vs temporal aggregation for helmet;
single-frame vs multi-frame consensus for OCR; association path A/B for triple
riding; tracker (ByteTrack vs OC-SORT) A/B where it affects violation metrics.
Ablations vary one factor under otherwise identical conditions.

## 19. System throughput and latency
Report **throughput** (realtime factor / frames-per-second-equivalent, measured),
**end-to-end latency**, **per-stage latency** where measurable, and **P50/P95**
latencies. Numbers are reported only **after measurement**; no FPS or real-time
claim is made in advance.

## 20. Resource-usage reporting
Report **CPU usage**, **GPU usage**, **VRAM**, **RAM**, alongside **video
resolution**, **frame rate**, **batch size**, and a **hardware description**.
Distinguish **offline batch processing** from the **near-real-time demo mode**
(detector + tracker + rules on one stream); real-time is never claimed for the
full concurrent stack.

## 21. Human-review metrics
Report **reviewer approval rate**, **rejection rate**, **correction/relabel
rate**, **review latency**, **inter-reviewer disagreement rate** (if multiple
reviewers), **reason-code distribution**, and **per-violation acceptance rate**.
The review UI is not implemented in U4.

## 22. Robustness slices
Evaluate on slices where metadata supports them: **daylight, dusk, rain, blur,
occlusion, crowd density, object size, distance/perspective, camera viewpoint**.
Slices are reported separately, not averaged away.

## 23. Night-operation treatment
Night remains a **robustness-analysis slice only**, **not** a claimed supported
operating condition, unless later evidence explicitly changes project scope.
Night ANPR without IR is treated as out of supported scope.

## 24. Failure-analysis requirements
Every layer includes a **qualitative failure catalog** (representative clips/
crops with root-cause notes) and, for events, per-slice dissociation analysis.
Negative and tie results are reported honestly, not hidden.

## 25. Reporting requirements
Each result carries its provenance (§4), the split/holdout used, the operating
thresholds, confidence intervals or seed spread where practical, and explicit
statements of what was **not** evaluated. Implemented capability is always
distinguished from planned or simulated capability.

## 26. What constitutes a valid comparison
A comparison is valid only if the compared systems share: the **same test
split/holdout**, the **same ground truth and matching semantics**, the **same
metric definitions and operating thresholds**, and equivalent evaluation
conditions. Tuning on the test set, changing the matching threshold between
systems, or comparing across different holdouts invalidates the comparison. The
CNN-vs-ViT experiment (§8) is the canonical example of an enforced fair comparison.
