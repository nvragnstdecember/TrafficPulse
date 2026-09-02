# P4-U7 — detection-recovery protocol (frozen before any diagnostic number was produced)

- **Status:** Frozen 2026-09-01, before the diagnostic pass was run
- **Relationship to prior units:** P4-U5 (`p4u5-prereg`, commit `52839d0`) and P4-U6-V are
  **read-only inputs**. Neither is rerun, amended, or reinterpreted. P4-U6-V's population,
  crops, scores, frozen operating points, and `results_runtime_validation.json` stay
  exactly as they are; this unit writes to its own files only.

---

## 1. The question

P4-U6-V recovered **3,528 / 6,553** eligible single-rider riders (53.8%). The dominant
exclusion was `no_motorcycle_detected_at_iou_0.50` (2,690 = 41.0%). This unit asks *why*,
and whether recovery can be improved without an unacceptable detection-quality cost.

It does **not** ask which classifier is better. Detector recovery and classifier accuracy
are reported separately and never combined into a single "improvement" claim.

## 2. Baseline detector configuration (audited, unchanged)

| | |
|---|---|
| Checkpoint | `PekingU/rtdetr_r50vd` (`local_files_only=True`) |
| Family | DETR-style set prediction, 300 queries — **no NMS stage exists** |
| Input | `RTDetrImageProcessor`, resized to **640x640**, `do_pad=False` |
| Aspect handling | 1920x1080 -> 640x640 is **non-uniform**: x scaled 0.333, y scaled 0.593 |
| Rescale / normalise | `do_rescale=True` (1/255), `do_normalize=False` |
| Backend threshold | 0.50 (`post_process_object_detection`) |
| Adapter threshold | 0.50 (`DetectorConfig.score_threshold`, authoritative) |
| Class mapping | `serve.py` LABEL_MAP; `motorbike`(id 3) -> MOTORCYCLE, `person`(id 0) -> PERSON |
| Box handling | clipped to frame; zero-area boxes dropped |
| GT match | IoU >= 0.50, highest wins, each detection used at most once |

The only tunable operating point is the **score threshold**. There is no NMS/IoU knob to
adjust, and the 640x640 input is the resolution the checkpoint was trained at — changing it
is a different model configuration, not an operating point, and is out of scope here.

## 3. Failure taxonomy (kept distinct, never collapsed)

Every eligible rider lands in exactly one bucket:

- **A — no motorcycle detection at all.** No `motorbike` detection anywhere in the frame
  above the operating threshold that overlaps the GT box at all (best IoU = 0).
- **B — detected but unmatched.** A `motorbike` detection overlaps the GT box (IoU > 0) but
  the best IoU is below 0.50.
- **C — matched but association failed.** The motorcycle matched, but `associate_riders`
  linked zero riders, or more than one (violating the single-rider premise).
- **D — recovered**, subject to the head-crop gates.

A and B are further split by whether a *sub-threshold* detection existed, which is what
distinguishes an operating-point problem from a capability problem.

## 4. Diagnostic pass (one expensive run, all analysis offline)

RT-DETR is run **once** over the same 3,398 frames at a floor of **0.01**, recording every
`motorbike` and `person` detection with its score and box, alongside every annotated
motorcycle in that frame. Every threshold in section 5 is then evaluated offline from that
record, so no threshold is chosen by re-running the detector until a number looks good.

## 5. Threshold grid and selection rule (both pre-committed here)

**Grid, fixed:** `{0.50 (baseline), 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05}`.

**Selection rule, fixed:**

> Choose the threshold maximising **validation motorcycle-detection F1**, where a detection
> is a true positive if it matches an annotated motorcycle at IoU >= 0.50 (one-to-one,
> greedy by IoU) and a false positive otherwise, and every unmatched annotation is a false
> negative. Ties break to the **higher** threshold (the more conservative operating point).

This criterion is deliberately **independent of the classifier**. It cannot be gamed by
whichever threshold happens to flatter ResNet, DeiT, or zero-shot, and it balances recall
against precision rather than maximising recovery alone. Downstream crop recovery is
reported as a *consequence* of the selected point, never as the thing selected on.

**False-positive denominator.** Precision is measured against **all** annotated motorcycles
in each sampled frame — not merely the single-rider subset this project evaluates — because
a detection of a genuine multi-rider motorcycle is not a detector error. The residual
caveat is recorded in section 7.

## 6. Test discipline

Validation selects; test is read once, afterwards. If the selected threshold differs from
the 0.50 baseline, the test population is re-derived at that threshold and the three
classifiers are re-scored on the new crops with **identical checkpoints and preprocessing**.
No model is retrained and no classifier threshold from P4-U6-V is re-selected: the frozen
`abstain_below` values (zero-shot None, ResNet 0.80, DeiT 0.60) are reused verbatim.

Coverage and accuracy are reported separately. **An increase in recovered crops is not an
accuracy claim**, and a macro-F1 measured over a different (larger) crop population is not
directly comparable to P4-U6-V's — where they are compared, the shared subset is used.

## 7. Known limitations of this design, stated in advance

- HELMET annotates motorcycles **with tracked riders**. A parked, occluded, or untracked
  motorcycle may be genuinely present and unannotated, so measured precision is a **lower
  bound** and the false-positive count an **upper bound**.
- The corpus samples frames (stride 5) and caps crops per track, so frame-level annotation
  is complete but the *eligible* population is a sample of it. Detection metrics use the
  full frame annotation; recovery metrics use the eligible population. The two denominators
  are different by construction and are always labelled.
- Occlusion is not annotated in HELMET. Any occlusion statement here is inferred from
  box overlap between annotated motorcycles, and is described as such.

## 8. Stop conditions

Work stops and reports rather than proceeding if: recall gains require a false-positive
explosion; the fix would need retraining or a different checkpoint; the frozen data cannot
support a defensible validation; or the change would alter what P4-U6-V measured.
