# TrafficPulse Label Ontology

**Version:** 1.0.0 · **Status:** frozen (Phase 0-F, U3)
**Canonical machine-readable source:** [`configs/ontology.yaml`](../configs/ontology.yaml)
**Contract source of truth for closed sets:** `src/trafficpulse/contracts/enums.py`

This document is the human interpretation and annotation guide. The machine-readable
registry is `configs/ontology.yaml`; where the two ever disagree, the YAML is the
canonical identifier registry and this document is the explanatory companion.
Closed identifier sets (helmet states, signal states, object classes, violation
types, observation types) are kept exactly consistent with the frozen U2 contract
enums by the tests in `tests/ontology/`.

---

## 1. Purpose and status

The ontology freezes the *meaning* of TrafficPulse labels **before** any dataset
preparation, annotation, model training, or evaluation begins. Annotation is
costly and effectively irreversible, so the label vocabulary and its semantics
must be fixed first.

This ontology defines **semantics only**. It contains no rule thresholds,
durations, grace periods, speed limits, confidence cutoffs, or executable logic —
those belong to later rule/configuration layers (U4/U5/Phase 1). Version 1.0.0 is
the initial frozen ontology.

## 2. Ontology principles

- **Label observable visual facts, not assumptions.** Do not infer invisible or
  hidden states.
- **Prefer abstention over guessing.** Where the ontology provides an
  abstention/unknown label, use it when evidence is insufficient.
- **A single-frame classification is never a confirmed violation.** Perception
  produces observations; deterministic rules later accumulate evidence over time
  and confirm events. These layers are separate.
- **Classifier labels ≠ final legal/policy decisions.** The visual label records
  what is seen; enforcement/policy meaning is decided in the rule layer.
- **Closed sets mirror the contracts.** Any closed identifier set here matches the
  corresponding U2 enum exactly.
- **Preserve source grouping** (camera/video identity) so later dataset splitting
  can be leakage-safe.

## 3. Helmet-state annotation guide

Canonical states (`helmet_states` in the YAML), matching the U2 `HelmetState`
enum: `helmet`, `no_helmet`, `turban`, `uncertain`.

| Label | Annotate when… |
|---|---|
| `helmet` | A qualifying protective helmet is visibly present on the rider's head with sufficient evidence. |
| `no_helmet` | The head region is sufficiently visible and a qualifying protective helmet is visibly absent. |
| `turban` | The rider is visibly wearing a turban. A **distinct** label — see §4. |
| `uncertain` | Evidence is insufficient for a reliable decision — an **abstention**, not a violation. |

Annotation notes:

- Label the **head region per rider slot** (driver / pillion / third), not the
  whole motorcycle.
- The helmet label is a **visual classification**, not a legal determination.
- `no_helmet` on a single frame is **not** a confirmed no-helmet event (see §8, §11).

## 4. Helmet ambiguity and abstention policy

- **`turban` is a distinct visual label.** It must **never** be silently merged
  into `no_helmet`. In the YAML each helmet state carries
  `automatic_violation_semantics: none`, and `turban` additionally carries
  `distinct_from_no_helmet: true`.
- **`turban` does not imply a violation.** The label records only what is seen. It
  asserts **no** legal exemption and **no** violation. Whether a turban should be
  treated as exempt, as requiring abstention, or otherwise is a **downstream
  project-policy / rule-layer decision**, subject to verification of the
  applicable regulation — it is **not** decided by this ontology, and this
  document makes **no** legal claim about it.
- **`uncertain` abstains.** It is an explicit abstention state (`abstains: true`
  in the YAML) and must **not** be treated as `no_helmet`. Uncertainty can arise
  from blur, occlusion, tiny crop size, truncation, poor illumination, ambiguous
  headgear, or otherwise insufficient visible evidence.
- **Prefer `uncertain` over guessing.** If you cannot reliably tell, abstain.

## 5. Signal-state semantics

Canonical states (`signal_states`), matching the U2 `SignalState` enum:

| Label | Meaning | Visibility |
|---|---|---|
| `red` | Signal head showing a red (stop) aspect. | classified |
| `amber` | Signal head showing an amber (caution) aspect. | classified |
| `green` | Signal head showing a green (go) aspect. | classified |
| `off` | Signal head visibly not showing an active aspect (dark/blank/non-operating). | classified |
| `unknown` | State could not be reliably determined (occluded / out of frame / low confidence). | indeterminate |

`red`, `amber`, `green`, and `off` are **determinate** observed states.
`unknown` is **indeterminate**: downstream reasoning should abstain rather than
assume an aspect.

## 6. Object-class semantics

Closed detector/tracker classes (`object_classes`), matching the U2 `ObjectClass`
enum: `motorcycle`, `car`, `bus`, `truck`, `auto_rickshaw`, `bicycle`, `person`,
`license_plate`.

- `motorcycle` is the primary subject of helmet, triple-riding, and plate
  reasoning.
- `person` covers any human (pedestrian or occupant/rider); rider-to-vehicle
  association is a separate concern, not an object class.
- `license_plate` is a plate **region** (subject of ANPR), not a vehicle.
- Ambiguous or out-of-vocabulary objects are handled by **not** forcing them into
  a class; the set is closed and is not expanded speculatively. Expanding it is a
  versioned ontology + contract change (see §12).

## 7. Violation-type definitions

The six locked violation types (`violation_types`), matching the U2
`ViolationType` enum. For each: the machine id, a concise meaning, and the
conceptual upstream observations it may consume later. **No thresholds or rule
logic are defined here.**

| Id | Meaning | Conceptual upstream observations |
|---|---|---|
| `no_helmet` | Motorcycle rider riding without a qualifying helmet. | `helmet_state`, `rider_count` |
| `triple_riding` | Three or more riders on one motorcycle. | `rider_count` |
| `red_light_jumping` | Vehicle crossing the stop line into the junction on red. | `signal_state`, `in_zone` |
| `wrong_way` | Vehicle travelling against the legal lane/road direction. | `heading_vs_lane` |
| `illegal_stopping` | Vehicle stopped/parked in a configured no-stopping/parking zone. | `stationary`, `in_zone` |
| `speeding` | Vehicle exceeding the applicable limit in a calibrated zone (feasibility-gated). | `speed` |

Each violation also has documented abstention/ambiguity conditions in the YAML
(`abstention_conditions`). The ontology defines **semantics**, not rule
thresholds, temporal durations, grace periods, speed limits, or implementation
logic.

## 8. Observation-type definitions

The seven per-frame observation types (`observation_types`), matching the U2
observation discriminators (`obs_type`) exactly. Each states what fact it
represents, what it does **not** imply alone, and which violation families may
consume it.

| Id | Represents | Does **not** imply by itself | May be consumed by |
|---|---|---|---|
| `in_zone` | Whether a track occupies a configured zone in a frame. | That a violation occurred. | `red_light_jumping`, `illegal_stopping` |
| `signal_state` | The observed signal aspect for a light ROI. | A violation; it is not tied to any vehicle. | `red_light_jumping` |
| `heading_vs_lane` | A track's heading vs the lane's legal direction. | A wrong-way violation (single frame). | `wrong_way` |
| `stationary` | Whether a track is stationary (with dwell so far). | Illegal stopping. | `illegal_stopping` |
| `rider_count` | Estimated riders on a motorcycle in a frame. | A confirmed triple-riding event. | `triple_riding` |
| `helmet_state` | Per-rider-slot helmet state for a head region. | A confirmed no-helmet event (even when `no_helmet`). | `no_helmet` |
| `speed` | Ground-plane speed with explicit uncertainty. | Speeding without calibration quality and rule context. | `speeding` |

## 9. Frame-level labels versus temporal/event-level ground truth

- **Frame-level labels** describe a single frame/crop (e.g. a `helmet_state` for
  one head, a `rider_count` for one bike). They are the annotation and
  classification unit.
- **Event-level ground truth** describes a confirmed violation over time (a track,
  an interval, a rule outcome). It is produced by temporal reasoning and is a
  separate annotation/ground-truth product.
- Keep the two **separate**. A frame label is an input to reasoning, never itself
  an event. Do **not** use future frames to label a current frame unless a dataset
  task explicitly defines clip-level/temporal annotation.

## 10. Annotation edge cases

- **Occlusion:** if the region needed for a label is occluded, use the
  abstention/unknown label rather than guessing the hidden state.
- **Truncation:** if the region is cut off by the frame edge such that the label
  cannot be reliably determined, abstain.
- **Tiny / blurry crops:** for helmet state, small or motion-blurred head crops
  are `uncertain`.
- **Ambiguous headgear:** caps, hoods, scarves, or unclear coverings that are not
  clearly a qualifying helmet or a turban are `uncertain`, not `no_helmet`.
- **Borderline / contested cases:** escalate for review or exclude; for this
  ontology version, contested closed-class cases fold into the abstention label
  rather than being guessed.

## 11. Relationship between classifier outputs and violation decisions

Classification and violation confirmation are **deliberately separate**:

1. A model emits a **classification** (e.g. `helmet_state = no_helmet`) for a
   frame — a perception fact with its own confidence.
2. Rules later **accumulate** such observations over time, apply scene context and
   explicit criteria, and only then **confirm** an event.
3. Confirmed events undergo **human review** before any simulated penalty.

Therefore a classifier label is **not** a violation, and the ontology encodes this
by giving every helmet state `automatic_violation_semantics: none`. The mapping
from labels to policy outcomes (including how `turban` and `uncertain` are
treated) lives in the rule layer, not in this ontology.

## 12. Versioning and change-control policy

- The ontology carries an explicit semantic version (`ontology.version`,
  currently **1.0.0**).
- Any future semantic change (adding/removing/renaming a label, or changing a
  label's meaning) **requires**:
  - an ontology **version change**;
  - review of **affected datasets** already annotated under the old semantics;
  - review of **affected model outputs** trained/evaluated under the old
    semantics;
  - review of **evaluation compatibility**;
  - review of **rule-engine consumers** of the changed labels.
- Closed identifier sets must remain consistent with the U2 contract enums; a
  divergence is a deliberate, reviewed contract change, not an edit here alone.
- Migration tooling is intentionally **not** part of this unit.
