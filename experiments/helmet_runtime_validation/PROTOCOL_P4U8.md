# P4-U8 — runtime-recovery re-measurement (evaluation-scaffold correction)

- **Status:** Frozen 2026-09-01, before any corrected recovery rate was computed.
- **Relationship to prior units:** P4-U5 (`p4u5-prereg`, commit `52839d0`), P4-U6-V and
  P4-U7 are **read-only inputs**. None is rerun, amended, or reinterpreted. Their files
  (`experiments/helmet_cnn_vit/**`, `runs/helmet_cnn_vit/**`,
  `runs/helmet_runtime_validation/**`, `runs/helmet_detection_recovery/**`) are never
  written to by this unit, which writes only to `runs/helmet_runtime_validation_p4u8/`.

---

## 1. What this unit corrects, and what it does not claim

P4-U7 established that most of P4-U6-V's apparent detection failures were a **box-convention
mismatch between the annotation and the runtime**, not a detector failure:

- a HELMET annotation row is **one box per tracked motorcycle**, and that box encloses the
  **motorcycle together with its riders** (`helmet_cnn_vit.labels`: one row carries the
  helmet state of every rider on the bike);
- RT-DETR's `motorbike` box encloses the **vehicle only**.

Comparing those two boxes by IoU compares different objects. The measured recovery rate was
therefore an artefact of the evaluation scaffold.

This unit replaces the scaffold's matching rule with one that compares like with like. The
difference between P4-U6-V's recovery and P4-U8's is an **evaluation-scaffold correction**.
It is **not** a detector improvement: the detector, its checkpoint, its input geometry, its
score threshold, the tracker, the association policy and the head-crop geometry are all
identical to P4-U6-V. Nothing about the production system changes in this unit, and no
production file is edited by it.

## 2. The runtime is not the scaffold

Stated explicitly, because conflating the two is the error being corrected.

**Runtime path (unchanged, and needs no ground truth at all):**

    frame -> RTDetrDetector -> DetectionAdapter -> IouTracker -> associate_riders
          -> extract_head_region(0.30) -> min_crop_height_px gate -> classifier

The classifier's crop is the top 30% of the **person** box. Producing it involves no
annotation and no matching. A deployed TrafficPulse never performs the step in section 3.

**Evaluation scaffold (this unit's only subject):** deciding whether a runtime
motorcycle-and-rider corresponds to a particular frozen HELMET annotation, so that the
annotation's label may be attached to the crop the runtime produced.

## 3. The matching rule (frozen here, before any corrected number)

### 3.1 Rule

For each frame, at the **unchanged** production operating point (score threshold 0.50):

1. Take every `motorbike` detection and every `person` detection the adapter emits.
2. Run the **production** `associate_riders` over them (`RiderAssociationConfig` defaults:
   IoMin >= 0.30, each person assigned to at most one motorcycle, a motorcycle may carry
   several). No association logic is reimplemented.
3. For each detected motorcycle *m*, its **evaluation proxy** is the axis-aligned union

       proxy(m) = union( box(m), box(r) for every rider r associated to m )

   A motorcycle with no associated rider has `proxy(m) = box(m)`.
4. Match each eligible annotation to a proxy by **IoU >= 0.50**, highest IoU wins, each
   proxy used at most once per frame. Annotations are considered in `crop_id` order and
   take the best still-unused proxy — **the identical assignment procedure P4-U6-V used**,
   so the box convention is the only variable that changes.
5. The crop's rider is the person box associated to the matched motorcycle. **Exactly one**
   associated rider is required; zero or two-or-more is a recovery failure with its own
   reason code, never a silent drop.

### 3.2 Every property the rule is required to have

- **Deterministic.** Pure geometry over a fixed detection record, with a fixed iteration
  order and a fixed tie-break. No randomness, no wall-clock, no model state.
- **Independent of classifier predictions.** No classifier is loaded, called, or consulted
  anywhere in this unit. The rule cannot be gamed by whichever backend it would flatter.
- **Applied identically to val and test.** One code path, one constant, no per-split branch.
- **Frozen before test evaluation.** This document is written before the corrected recovery
  is computed on either split.
- **No free parameter to tune.** The IoU floor stays at **0.50**, inherited verbatim from
  P4-U6-V section 3 and from COCO convention. The association threshold stays at the
  production default. Nothing in this unit is selected on data, so there is no quantity
  that *could* be tuned on test. Validation is used as a **sanity check**, not a selector.

### 3.3 Why the union proxy, and what was rejected

The recommended construction is adopted, because inspecting the repository produced no
better existing representation: `DetectionAdapter` performs no grouping, `IouTracker`
alters no geometry (a fresh per-frame tracker assigns identity only, and `tainted` is
always `False`), `associate_riders` is the *only* component in the codebase that groups a
rider with a vehicle, and P4-U7's `recovery.classify_frame` compares motorcycle-only boxes,
which is exactly the convention being corrected.

Rejected alternatives, and why:

- **Lower the IoU floor against the motorcycle-only box.** Rejected: it buys agreement by
  weakening the criterion rather than by fixing the correspondence, and it introduces a
  parameter that would then have to be selected on data.
- **Shrink the annotation to a motorcycle-only estimate.** Rejected: HELMET carries no
  vehicle-only box. Deriving one would be synthesising an annotation the dataset does not
  contain.
- **Containment measures (IoMin, centre-in-box) against the motorcycle-only box.**
  Rejected: they are satisfied by any sufficiently small box inside the annotation — a
  wheel, a pillion, a pedestrian — so they weaken one-to-one matching in exactly the cases
  that matter.
- **Union with *all* person boxes overlapping the motorcycle, rather than the associated
  ones.** Rejected: it would use a grouping the runtime does not perform, and it silently
  repairs association failures instead of counting them. (P4-U7's offline replication used
  this looser form; section 6 records the discrepancy.)

### 3.4 A stated risk, and the diagnostics that bound it

Enlarging the runtime box can only raise IoU against an annotation that is itself large, so
the rule could in principle manufacture matches. Three things bound that:

1. matching stays **one-to-one** with the **unchanged 0.50** floor;
2. a **suspicious-match** count is reported: matches whose *motorcycle-only* IoU with the
   annotation is below 0.10, i.e. where the union did essentially all the work;
3. a **contested-proxy** count is reported: proxies for which more than one annotation
   clears 0.50, i.e. where the assignment order could have changed the outcome.

Both diagnostics are published whatever they say.

## 4. Population (unchanged from P4-U6-V)

**Single-rider motorcycles only** (`rider_count == 1` in the frozen corpus), because
`observations.helmet.rider_slot` returns `DRIVER` only for exactly one associated rider and
`UNKNOWN` otherwise — the shipped tracker supplies no velocity, so driver-versus-pillion
cannot be read off image-space position. Multi-rider annotations are **excluded and
counted**, never discarded silently, and their share of the frozen population is reported.

No driver/pillion attribution is invented anywhere in this unit.

## 5. Reported quantities

For **val** and **test** separately, and never pooled:

- total frozen crops; single-rider eligible; eligible share;
- recovered count and recovery rate;
- exclusions by reason, in the P4-U7 taxonomy (A nothing overlapping / B detected but below
  the IoU floor / C matched but association failed / D recovered), plus the head-crop gates;
- class distribution of the recovered population and of the eligible population;
- crop-height distribution (min / quartiles / max) of the recovered head regions;
- recovery by crop-size quartile (quartiles cut on the **eligible** annotation areas of the
  split being reported, so the strata are a property of the population, not of the result);
- recovery by scene/site;
- recovery by rider count (over the full frozen population, single- and multi-rider);
- recovery by class;
- association-failure rate and detection-failure rate, each with its denominator stated.

Every P4-U8 number is published beside its P4-U6-V counterpart and labelled
**evaluation-scaffold correction**.

## 6. Fidelity of the offline reconstruction (and how it is proved)

The detector is **not** re-run. P4-U7's `detection_dump.jsonl` already records every
`motorbike` and `person` detection at a 0.01 floor over exactly the 3,398 frames P4-U6-V
processed; filtering it at 0.50 reproduces the production detection set exactly, because
RT-DETR's score threshold is a post-processing filter over a fixed set of 300 query outputs
and the adapter's clipping is deterministic.

That claim is not asserted, it is **tested**: running this unit's matcher in
`motorcycle_only` mode must reproduce P4-U6-V's recovered crop-id set **exactly**
(3,528 crops, and the same per-reason counts). If it does not, the reconstruction is wrong
and no corrected number may be published. This check runs before any corrected number is
computed, and its outcome is recorded in the results file.

One residual difference is recorded rather than hidden: the dump stores motorcycles and
persons in separate lists, so the interleaved detection order that determines
`IouTracker`'s `trk-N` numbering is not recoverable. Track ids therefore differ in name from
a live run. They affect only `associate_riders`' tie-break between two motorcycles a rider
overlaps by an **exactly equal** float ratio; the exact-reproduction check above is what
demonstrates this is not operating in practice.

## 7. What this unit deliberately does not do

- It does **not** rerun classifier scoring. Whether the corrected population justifies a
  re-scoring is a separate question, answered in P4-U9 under its own protocol.
- It does **not** change the detector, its threshold, its input geometry, the tracker, the
  association policy, or the head-crop geometry.
- It does **not** repair the genuine residual issues P4-U7 identified (small objects,
  Pakokku_urban clustering, association failures, class-biased recovery, the non-uniform
  640x640 input). It re-measures them under the corrected convention and reports them.
- It makes **no** adoption claim about any classifier backend.
