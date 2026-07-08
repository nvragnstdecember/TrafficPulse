# TrafficPulse — Architecture & Feasibility Review

**Status:** Accepted as the canonical architecture reference (living document)
**Supersedes:** none
**Relationship to spec:** Interprets and constrains `TRAFFICPULSE_MASTER_SPEC.md`; does not modify it. Where this document and the spec diverge, the divergence is listed explicitly in §27 and routed to an ADR or a spec-revision note.
**Verification legend:** [V] verified against a primary source during review · [K] training-knowledge, re-verify before reliance · [R] recommendation (not yet a decision) · [A] assumption · [U] unresolved · [E] requires later empirical validation.

---

## 1. Purpose & status

This is a durable architecture decision document. It records what has been decided, what remains a recommendation, and what is unresolved, so that an implementation agent can act without re-deriving the reasoning. It is not a transcript. It preserves uncertainty deliberately: estimates are not promoted to facts, unconfirmed dataset access stays unconfirmed, and no accuracy or throughput number is stated as measured.

The review it summarizes reached a **GO WITH CHANGES** recommendation (§27). Nothing here overturns the locked scope, the approved principles, or the approved six-unit Phase 0-F plan; several items sharpen them, and one spec gap (evaluation-footage acquisition) is elevated to a first-class dependency.

## 2. Project thesis

TrafficPulse is an evidence-first, multi-violation traffic-detection and simulated-penalty system for fixed-camera roadside video in Indian urban traffic. Its central commitment: **a single model output or single-frame detection is never a violation.** Entities are tracked over time; typed observations are derived; scene geometry and explicit rules are applied; evidence accumulates; confirmed events become reviewable cases; only a human-approved case yields a *simulated* penalty.

Processing is **offline-first** (recorded video in, events out) with a **reduced near-real-time demo mode**. "Real-time" is never claimed for the full concurrent stack (see §22).

## 3. Scope — six locked violations

1. No-helmet riding
2. Triple riding
3. Red-light jumping
4. Wrong-way driving
5. Illegal stopping / parking
6. Speeding (feasibility-gated — §17)

Accident / stalled-vehicle detection is **stretch-only** and must not displace or delay the six. Scope is not silently expanded or reduced.

## 4. Critical review of the master specification

**Strong decisions to preserve unchanged:** the ten principles (spec §3); the pre-gated speeding scope; the ViT anti-branding clause with a single mandatory experiment and the second comparison explicitly optional; the research-first anti-scaffold workflow; dataset-registry duties and "no download before licence review"; the leakage section; hardware realism; the honest 10–14-day framing; the §26 guardrails.

**Assumptions requiring validation:** existence of usable-licence public helmet/rider data [U]; ability to secure a recording site with a visible signal head, permission, and safe elevated placement [U]; adequacy of pretrained-detector transfer to Indian fixed-camera viewpoints [E]; sufficiency of 16 GB system RAM under concurrent decode + dataloading [E].

**Contradictions / ambiguities resolved here:** the external "real-time" tagline overstates the spec — align the tagline to the spec, not vice versa (§22, §27). Spec §2.2 hard-codes a *mechanism* for triple riding; this review treats it as a *capability* (§5d, §27). "Modern CPU ~4 GHz" is untestable but harmless.

**Harder than they appear:** rider–vehicle association in dense traffic; track persistence for minutes-long stationary vehicles; congestion suppression for the stopping rule; night signal-state classification; two-line motorcycle plates; event-level ground-truth annotation cost.

**Over-engineered:** two separate dashboards (merge into one analytics/evaluation view) [R]; "experiment tracking support" (keep file-based) [R].

**Missing technical concerns (now addressed):** evaluation-footage acquisition plan (§9, §26 risk #1) — the single largest omission; PTS / variable-frame-rate discipline as an *ingestion-wide* requirement, not speed-local (§17); framework/model licence posture (ADR-001, §11); religious-headwear exemption as an *ontology* requirement (§5e); red-light rule semantics (permitted turns, amber grace, encroachment vs junction entry) (§5b); reprocessing / event-identity semantics (ADR-004, §19); absent team size/roles making schedule discipline unquantifiable.

**Scope to defer:** night as a *supported condition* (keep as robustness analysis only, §23); any detector-family CNN-vs-ViT comparison (optional, gated); accident/stalled detection (stretch).

## 5. Capability decomposition

Shared perception core for all six: PTS-timestamped frames → multi-class detector (motorcycle, car, bus, truck, auto-rickshaw, bicycle, person, plate) → detection-based tracker (no re-ID needed for a fixed camera) → per-track state → typed observations → rules. Rules consume **only** observations.

### 5a. Wrong-way driving — *recommended first end-to-end violation*
Track heading/displacement inside a lane polygon with a configured legal angular range → observation `heading_vs_lane` → sustained contradiction (deviation > θ_max for ≥ N frames AND cumulative displacement ≥ threshold) → hysteresis-confirmed event. Learned: detector only. Abstain: short track, or near the polygon boundary, or tainted (ID-switch) tracks. FP: ID switches, U-turns/reversing (needs exclusion zones). Feasibility: excellent; difficulty low; demo reliability high.

### 5b. Red-light jumping
Vehicle tracks + signal-state observation (ROI classifier over a configured light-head ROI, **or** an external signal log — the spec's "source or recognition" abstraction is correct). Per-track FSM: crossing the stop line into the junction conflict zone while `signal_state = RED` beyond an amber-onset grace, excluding configured permitted movements (e.g., a signed free left). Learned: detector + tiny signal classifier (HSV baseline acceptable first). Abstain: signal ROI occluded/low-confidence, crossing within a state-transition uncertainty window. Feasibility: high **iff** the site's signal head is visible [U]. Difficulty medium; demo reliability high (daytime, good site).

### 5c. Illegal stopping / parking
`stationary` observation (speed < ε with hysteresis) inside a no-stopping polygon → dwell timer with max-gap tolerance → violation when dwell > T AND congestion-suppression passes (scene-level flow state). Risk: tracker ID churn on long-stationary objects; mitigation is a **stationary-object memory** re-associating a "new" track at the same location (IoU ≥ τ) with the prior dwell record. Learned: detector only. Abstain: congestion flag active, dwell interrupted beyond tolerance. Difficulty medium; demo reliability medium-high.

### 5d. Triple riding — capability, not fixed mechanism
Per-motorcycle **rider-count estimation**, via either (i) person-detection + geometric association, or (ii) a rider-occupancy classifier on the motorcycle crop (1/2/3+). In dense traffic (ii) is typically more robust and directly trainable from AI City T5-style labels **if access is granted** [U][R]. Observation `rider_count(track, t, conf)` → sustained count ≥ 3 over K of N good-visibility frames. Abstain: overlapping bikes, oscillating count, partially visible child (flag for review). Difficulty medium-high; demo reliability medium (camera-angle-dependent).

### 5e. No-helmet riding — hosts the mandatory CNN-vs-ViT experiment
Motorcycle track → rider slots (driver / pillion ordering) → per-slot head crop → helmet-state classifier → observation `helmet_state(track, slot, t, conf, crop_quality)` → quality-weighted temporal aggregation → confirmed event. **Ontology requirement:** annotation labels {helmet, no_helmet, turban, uncertain}; at the rule layer `turban → exempt (no event)`, `uncertain → abstain`. (Reflects the Motor Vehicles Act §129 turban proviso [K — verify exact provision before printing].) Abstain: head crop < ~24 px, motion blur, slot instability. FP: dark hair vs black helmet at low res, caps/hoods/scarves, helmet present-but-not-worn. Difficulty medium-high; demo reliability medium (resolution-dependent); temporal aggregation is what makes it demoable.

### 5f. Speeding — gated experiment only (§17)
Homography → smoothed ground-plane trajectory → `speed(track, t, σ)` computed only inside a calibrated measurement zone using PTS deltas → zone-median vs limit + uncertainty margin, hypothesis only if `v_est − k·σ > limit`. Learned: none beyond detector. Feasibility: as a constrained calibrated-zone experiment only. Difficulty high (validation, not code); demo reliability low-medium; excluded from penalty simulation until the gate passes.

## 6. Violation feasibility matrix

| Violation | Extra learned components | Core reasoning | Difficulty | Demo reliability | Verdict |
|---|---|---|---|---|---|
| Wrong-way | none beyond detector | geometry + hysteresis | low | high | first rule, Phase 1 |
| Red-light | tiny signal classifier | signal × line-crossing FSM | medium | high (site-dependent) | core, Phase 2 |
| Illegal stopping | none | dwell + zones + congestion guard | medium | medium-high | core, Phase 2 |
| Triple riding | rider-count estimator | sustained count | medium-high | medium (angle-dependent) | core, Phase 3 |
| No-helmet | helmet classifier (+ localization) | per-slot temporal vote | medium-high | medium | core, Phase 3; hosts experiment |
| Speeding | none | homography + uncertainty | high (validation) | low-medium | gated experiment only |

## 7. Dataset strategy & candidate matrix

No single all-in-one dataset. A dataset **registry** records provenance, licence, access status, split policy, and checksums per learned capability; nothing is downloaded before its registry entry is reviewed. Kaggle/Roboflow mirrors are not treated as primary sources; where a mirror is the only path, the entry records the provenance risk.

All entries [K] unless marked. Access reflects training-cutoff knowledge and **must be re-verified from the official source** before the registry marks anything usable.

| Dataset | Origin | Access | Licence | Modality / scale | Proposed use | Label fit |
|---|---|---|---|---|---|---|
| COCO 2017 | cocodataset.org | Open | annotations CC-BY 4.0; images mixed | 118k images | pretrained weights only | direct for pretraining |
| IDD | IIIT-H + Intel (WACV'19) | Registration-gated [K] | research/non-commercial [K—verify] | ~10k+ seg; detection subset [K—verify] | appearance-domain detector fine-tune | direct labels; viewpoint mismatch (dashcam) |
| AI City 2023 Track 5 | NVIDIA AI City Challenge | Data-agreement gated; no redistribution [U] | challenge agreement, research | ~100 short CCTV-ish videos [K—verify] | helmet crops **and** rider-count **and** detection fine-tune | direct for both — **highest value** |
| HELMET (Myanmar) | Siebert & Lin (~2020) | Verify hosting [K] | research [K] | tens of thousands of frames [K—verify] | helmet classifier pretraining/augmentation | derived crops needed |
| UA-DETRAC | Univ. at Albany | Flaky availability [U] | research | 100 seq, ~10 h | tracker smoke tests only | motorcycles not annotated — disqualifying gap |
| MOT17/20 | MOTChallenge | Open reg. | CC-BY-NC-SA | pedestrian video | tracker integration sanity only | not our classes |
| S2TLD / BSTLD / LISA | SJTU / Bosch / UCSD | mixed [K] | mixed non-commercial [K] | traffic-light frames | optional signal-classifier pretraining; night reference | marginal — ROI formulation barely needs |
| CCPD | Xu et al. (ECCV'18) | Open | **MIT [V]** | ~250k+ images | plate-**detector** pretraining only | Chinese OCR labels not applicable |
| Synthetic Indian plates | self-generated | N/A | self-owned | unlimited | primary OCR training | direct, controllable, 1- and 2-line |
| BrnoCompSpeed | Brno UT (~2017) | Request-based [U] | research agreement | ~6 sites, fixed-camera, reference speeds | validate speed **method** before trusting own calibration | direct for method validation |
| Own fixed-camera footage | self-collected w/ permission | **to be secured [U]** | self-owned + ethics process | target ≥ 4–6 h, ≥ 2 sites | **all final evaluation**; fine-tuning source | direct by construction |

Load-bearing dependencies: **AI City T5** (changes annotation budget ~2–3×) and **own footage** (the evaluation substrate for event-level metrics, ANPR, and speed).

## 8. Dataset gaps & custom-data requirements

Split unit = **source video** (minimum), with whole-site holdout for the final generalization claim; derived crops inherit their source video's split; no track spans splits; thresholds tuned on validation only; splits generated deterministically from a seeded, committed manifest. Adjacent-frame leakage is the default failure mode of helmet-classification-on-video and is explicitly prevented.

Estimated custom annotation totals **~35–65 focused team-hours plus one field day** — planned as first-class work, not slack:

| Capability | Public sufficient? | Fine-tune? | Custom annotation | Est. effort | Split unit |
|---|---|---|---|---|---|
| General detection | bootstrap yes | probably (IDD + own) | 200–500 fixed-cam frames | 8–15 h | video/site |
| Helmet state | only if AI City granted [U] | yes | 500–1,500 head crops (4-label) | 4–8 h | video; crops inherit |
| Rider count | AI City-dependent | yes | 1–2k bike crops | 3–5 h | video; crops inherit |
| Signal state | marginal | tiny model | 300–1,000 ROI crops/site | 1–2 h | video + time-block |
| Tracking | none in-domain | training-free tracker | MOT GT 3–5 clips (semi-auto) | 10–20 h | clip |
| Rule GT (wrong-way/stopping/red-light) | N/A | N/A | event GT manifests | 2–4 h per video-hour | video/site |
| Plate detection | CCPD helps | yes | 300–600 plate boxes | 2–4 h | video |
| Indian OCR | none credible open [K] | synthetic + fine-tune | 300–500 real transcriptions (eval) | 3–5 h | video |
| Speed | Brno for method only [U] | N/A | 20–50 GNSS-logged passes | 1 field day | session |

## 9. Evaluation-footage acquisition — first-class dependency

This is elevated from a spec omission to a tracked project dependency. Own footage is the evaluation substrate for event-level metrics, ANPR, and speed, and its permissions/ethics timeline is the schedule's long pole (§26 risk #1). Requirements: institutional permission before recording; 2–3 scouted sites, at least one with an elevated view **and** a visible signal head; a written data-handling document; fallbacks (gated datasets + short opportunistic recordings with narrowed claims) if no site is secured by end of Phase 1. Decision point: if no site by end of Phase 1, re-scope the evaluation plan.

## 10. Model & method strategy

Licence column [V] = fetched from the official repository during review.

| Role | Candidate | Family | Licence | 8 GB fit | Windows | Notes |
|---|---|---|---|---|---|---|
| Detection | Ultralytics YOLO11/v8 (s/m) | CNN 1-stage | **AGPL-3.0 [V]** | yes (m@640, fp16) | excellent | fastest path; ADR-001 |
| Detection (permissive) | RT-DETR | hybrid DETR | **Apache-2.0 [V]** | R18/34 yes | good | NMS-free; more integration |
| Detection (permissive) | D-FINE | DETR-family | **Apache-2.0 [V]** | S/M yes | good | younger; fallback to RT-DETR if immature |
| Detection (avoid) | MMDetection | various | Apache | yes | mmcv build friction [K] | avoid on this machine [R] |
| Helmet CNN | ResNet-50 (timm) | CNN | **Apache-2.0 [V]** | trivial | excellent | ~25M params baseline |
| Helmet ViT | DeiT-Small (timm) | ViT | **Apache-2.0 [V]** | trivial | excellent | ~22M params |
| Optional pair | ConvNeXt-T / Swin-T (timm) | CNN / hier. ViT | **Apache-2.0 [V]** | yes | excellent | ablation only |
| Tracking | ByteTrack | detection-based MOT | **MIT [V]** | negligible | excellent | default; no re-ID for fixed cam |
| Tracking alt | OC-SORT | detection-based MOT | **MIT [V]** | negligible | excellent | easy A/B through occlusion |
| Tracking (avoid default) | BoxMOT | various | **AGPL-3.0 [V]** | — | — | licence-couples the stack; unnecessary |
| Signal state | MobileNetV3-small / HSV | tiny CNN / heuristic | Apache (timm) | trivial | excellent | start heuristic |
| OCR | PaddleOCR (PP-OCR) | det+rec | **Apache-2.0 [V]** | yes | run ONNX via onnxruntime [K] | strong default |
| OCR (trainable) | PARSeq | transformer STR | **Apache-2.0 [V]** | yes | good | fine-tune on synthetic Indian plates |
| OCR (baseline) | EasyOCR | CRNN-ish | **Apache-2.0 [V]** | yes | good | quick baseline; usually weaker [E] |
| Face redaction | YuNet (OpenCV Zoo) | tiny CNN | **MIT [V]** | trivial | excellent | adequate for redaction [E] |
| Tracking eval | TrackEval | metrics | **MIT [V]** | CPU | excellent | HOTA/IDF1/MOTA |
| Best-frame | heuristic composite | non-learned | N/A | negligible | excellent | size × sharpness × plate visibility; learned scorer unjustified [R] |
| Speed | homography + Kalman/SG | non-learned | N/A | negligible | excellent | see §17 |

**Compute reality [E, must be measured — none of these is a measured claim]:** all listed models train at the proposed sizes on 8 GB with fp16; detector + tracker + rules on a single 1080p stream is expected to support a near-real-time demo mode; the full concurrent stack (detector + helmet + occupancy + signal + ANPR) is not expected to run in real time — this is the evidence for offline-first (§22). **Windows [K]:** avoid mmcv; run PP-OCR via ONNX Runtime; PyAV for PTS-accurate decode; WSL2 acceptable for training only, demo stays native.

## 11. Licensing considerations & ADR-001 (resolved 2026-07-08 — see ADR-001)

> **Update 2026-07-08 (post-review):** ADR-001 is now **Accepted** — TrafficPulse
> adopts a **permissive-only** detector posture (**RT-DETR** primary direction,
> D-FINE alternative), with the detector behind the U2 `Detection` contract.
> Ultralytics' AGPL coupling of *trained weights* was re-verified from the official
> licensing page (2026-07-08) and confirmed, which strengthened the permissive
> decision. The analysis below is preserved as the Phase 0-R review record that led
> to that decision; read its present-tense "unresolved" wording as historical. See
> [ADR-001](adr/ADR-001.md).

Ultralytics is AGPL-3.0 [V] and effectively couples the repository to AGPL (Ultralytics' stated position is that trained weights inherit it [K]). The fully permissive alternative is RT-DETR/D-FINE (Apache [V]) + timm (Apache [V]) + ByteTrack/OC-SORT (MIT [V]) + PaddleOCR (Apache [V]).

**ADR-001 remains UNRESOLVED.** Recommendation for the decision (not yet decided) [R]: take Ultralytics for velocity, commit to an AGPL public repo, keep the detector behind the perception contract so a permissive migration stays bounded. This is a genuine team decision about future reuse.

**ADR-001 process (governs both this document and the Phase 0-F plan):**
- ADR-001 **may remain unresolved at the end of Phase 0-F** only if its unresolved status, a named **decision owner**, a **decision deadline**, and its **consequences** are explicitly recorded in the ADR file.
- ADR-001 **must be resolved before detector integration begins** in Phase 1.
- An unresolved ADR-001 **must not block unrelated, detector-independent Phase 1 work** — geometry utilities, synthetic track generation, rule-engine foundations, contracts-driven wiring, and other work that does not select or integrate a detector ecosystem may proceed while ADR-001 is open.
- The decision deadline is therefore "before the first detector-integration unit of Phase 1," and the ADR file names the owner responsible for meeting it.

## 12. Mandatory CNN-vs-ViT experiment

**Decision (locked):** the mandatory experiment is **helmet-state classification**. Rationale: it is the only comparison where architecture is the actual independent variable (a detector-family comparison confounds backbone with label assignment, NMS-vs-set-matching, and schedules); it fits compute (2 models × 3 seeds × ≤3 GPU-h is an overnight job); the outcome is genuinely uncertain in the low-data, small-blurry-crop regime; and classifier calibration feeds the temporal aggregator, giving the result system consequences. Any detector-family comparison is optional and gated behind the complete integrated system.

**Design (pre-register before training):**
- Task: 3-class (helmet / no_helmet / uncertain_occluded); annotation carries 4 labels (adds turban); turban becomes a 4th class only if ≥ ~150 samples/split, else folds into uncertain.
- Models: ResNet-50 vs DeiT-Small (timm, ImageNet-1k), 224², square-padded crops, native crop height recorded as covariate. Optional ablation pair: ConvNeXt-T vs Swin-T.
- Data: AI City T5 [U] + HELMET pretraining [K] + custom Indian crops; test set from held-out videos + whole-site holdout.
- Leakage: split unit = source video; crops inherit; no track spans splits; manifest frozen and committed before the first run; thresholds/selection on validation only.
- Augmentation fairness: shared base + each family's best-known recipe under an equal tuning budget (forcing identical recipes handicaps ViTs and is itself unfair).
- Tuning: ≤ 8 val-selected configs/family, ≤ 3 GPU-h/run, 3 seeds for the final config each. Total ≤ ~20 GPU-h.
- Imbalance: class-weighted CE (focal fallback); per-class metrics; no test-set rebalancing.
- Metrics: macro-F1 (primary), balanced accuracy, per-class P/R, PR-AUC(no_helmet), confusion matrices; calibration via ECE + reliability diagrams, temperature scaling on validation reported pre/post.
- Latency/VRAM: batch 1 and 32, fp16, `inference_mode`, 100 warmup, median of 1,000 timed iters; VRAM via `max_memory_allocated`; params + checkpoint size.
- Robustness: crop-height buckets (<32/32–64/>64 px); synthetic corruptions (blur, motion blur, JPEG, brightness) at three severities; day/night slices where counts permit.
- Statistics: shared test set; McNemar per seed pairing; mean ± std; a difference is claimed only if sign-consistent across all three seeds **and** a pooled bootstrap 95% CI on ΔmacroF1 excludes zero — otherwise reported as a tie interpreted through the accuracy–latency–VRAM tradeoff.
- Interpretation, pre-committed: negatives/ties reported as such; slice dissociations reported per-slice; interpretation references the data-scale literature, not post-hoc stories.

## 13. Temporal reasoning architecture

Layer boundaries as typed contracts (U2 deliverable). Evidence accumulates as a quality-weighted per-(track, rule) score (log-odds or EMA, chosen per rule, documented, unit-tested against synthetic tracks in Phase 1). Event lifecycle FSM:

`IDLE → CANDIDATE (score ≥ θ_enter) → CONFIRMED (score ≥ θ_hi sustained ≥ min_duration, observation gaps ≤ max_gap) → CLOSED`, with `ABSTAINED` a terminal alternative out of CANDIDATE.

Hysteresis via θ_enter/θ_exit separation; per-(track, rule) cooldown prevents event spam; deduplication merges within the cooldown window; the stopping rule additionally re-associates stationary tracks across ID churn. Track termination triggers a grace period, then open hypotheses close with an explicit disposition (confirmed-truncated / abstained-trackloss) — never silently dropped. ID-switch guards: kinematic continuity checks taint a track; **tainted tracks may abstain but never confirm.**

**Two guarantees to be built into the rule-engine base class (not convention) when it is implemented in Phase 1:** no rule emits CONFIRMED from fewer than two observations; every non-confirmation produces a logged, countable abstention (abstention is a metric). Confidence is stored as a component breakdown (detector, classifier, association, temporal consistency, geometric margin, calibration quality) and is never labeled a probability unless calibration is demonstrated.

## 14. Typed conceptual data flow

```
Detection → TrackState → Association → Observation → TemporalState
→ ViolationHypothesis → ConfirmedEvent → EvidencePackage → ReviewCase → SimulatedPenalty
```

- **Detection** — frame-level model output.
- **TrackState** — tracker-owned identity + kinematics.
- **Association** — rider↔vehicle links with confidence.
- **Observation** — typed per-frame derived fact (`in_zone`, `signal_state`, `heading_vs_lane`, `stationary`, `rider_count`, `helmet_state`, `speed±σ`). **The durable perception↔reasoning contract.**
- **TemporalState** — per-(track, rule) evidence accumulator.
- **ViolationHypothesis** — candidate under accumulation.
- **ConfirmedEvent** — immutable.
- **EvidencePackage** — manifest + hashed artifacts.
- **ReviewCase** — human workflow object.
- **SimulatedPenalty** — post-approval, simulation-only.

## 15. Observation-log boundary & deterministic replay

Because rules consume **only** Observations, the reasoning layer replays deterministically from the observation log — no GPU, no model, bit-exact. A reviewer or examiner can re-derive exactly why an event fired from the manifest plus the log. Perception replay is only approximately reproducible (GPU nondeterminism) and is deliberately **not** required for audit. **This separation is the project's strongest defensibility mechanism and is a preserved, accepted architecture decision.**

## 16. Scene configuration & calibration

One versioned YAML per camera/site, schema-validated (U5), hashed into every event: schema version; camera id + reference frame; image-space geometry (road/lane polygons with legal heading ranges, no-stopping zones, exclusion zones, junction conflict zone); stop-line segments; traffic-light ROIs + permitted-movements list (signed free left is a per-site flag, not an assumption [K]); homography (correspondences + matrix + quality metrics); measured reference distances; per-zone speed limits; per-rule thresholds; congestion-suppression parameters.

**Manual for the capstone:** click-based annotation over a reference frame; tape/wheel-measured distances; hand-drawn ROIs. **Explicitly future/production:** vanishing-point auto-calibration, learned lane extraction, signal-controller integration, drift detection. Calibration quality is stored data (reprojection RMSE, cross-checked distances, warn/abstain thresholds); rules read it and abstain when it is poor.

## 17. Speed estimation feasibility gate

**Verdict:** general monocular speed estimation is **not defensible** within these constraints; a **constrained calibrated-zone experiment** is, and is what the spec authorizes.

Error budget dominated by: calibration (planar-road assumption violated by crown/grade; control-point error propagates with distance); reference-point instability (bbox bottom-center jitter → large far-field error — mitigated by a near-field zone ~10–35 m, ≥ 8–10 observations, ground-plane smoothing, zone-median speed); timestamps (VFR common → `dt` from PTS only, ingestion-wide requirement, anomalous-dt segments rejected); tracking (occlusion/ID-switch → tainted tracks abstain).

Ground truth: GNSS-logged own-vehicle passes (Doppler steady-state ~±0.3–1 km/h [K]), 20–50 passes over 20–60 km/h, stopwatch cross-check over a ≥ 40 m baseline; BrnoCompSpeed validates the method if access is granted [U]. Reports MAE, MAPE, P95 abs error, signed bias, per band, with the empirical error distribution plotted.

**Candidate provisional gate — NOT a committed requirement.** The project has **not** committed to these numbers. They are **candidate provisional evaluation targets only**, and remain **pending justification in the U4 evaluation-protocol document**: speeding would participate in the integrated demo and penalty simulation only if held-out validation meets targets on the order of **MAE ≈ ≤ 3 km/h and P95 ≈ ≤ 6 km/h** in the 20–60 km/h band over ≥ 20 passes at the demo site. The exact numbers, bands, and pass counts are to be justified (or revised) in U4 before any reliance. Regardless of the final numbers: every reported speed is `v ± kσ` with σ propagated from pixel jitter and calibration RMSE, alongside the calibration version; if the gate fails, speeding is demoted to an experiment chapter (measured, analyzed, honestly negative) and excluded from penalty simulation; enforcement-grade language appears nowhere in either case.

## 18. Indian ANPR feasibility & evaluation

Formats to handle [K — verify against current MoRTH rules]: `SS RR X(X) NNNN`, BH series, HSRP IND mark, and — critically — **two-line layouts** (dominant on motorcycles, the vehicles four of six violations concern). Pipeline is **event-triggered** (spec §11): event → best-K frame selection (crop width × sharpness × frontal-ness) → plate detection → perspective rectification → **line split** (projection profile or PP-OCR's line detection) → per-line recognition (PaddleOCR baseline; PARSeq fine-tuned on synthetic Indian plates) → format-constrained decoding (state-code lexicon + regex) → multi-frame per-character weighted consensus → string + confidence; below threshold → abstain + reviewer manual-transcription field.

**Honest expectations to validate, not claims [E]:** daytime crops ≥ ~60 px wide → plausibly 50–75% exact-match; < 40 px → poor; night without IR → near zero (declare night ANPR out of supported scope). Evaluation: plate-detection P/R; end-to-end exact-match; character accuracy (1 − normalized edit distance); format-valid rate; single-frame vs consensus; bucketed by plate pixel-width and lighting; character-confidence calibration if feasible. Data: synthetic training + CCPD (MIT [V]) detector pretraining + 300–500 own-footage transcriptions for eval. Privacy: plate strings are personal data — stored only for target vehicles, non-target plates blurred, covered by retention.

## 19. Evidence engine

JSON manifest + content-addressed artifacts. Manifest: event id (ULID); rule id + version; camera/site id; scene-config hash; track id(s); timestamps (start/confirm/end, PTS + wall-clock); rule trace (ordered state transitions with inputs, thresholds, measured values, margins); confidence breakdown; artifact list (before/trigger/after frames, clip, trajectory, plate crop, per-frame + consensus OCR) each with SHA-256; model registry refs (name, version, weights hash); code version (git SHA); review state; audit-history ref; simulated-penalty state. Manifests are append-only; reprocessing with a new model version creates new event ids linked by (video, rule, time-window) lineage — **cross-run dedup semantics are ADR-004 (may remain proposed).** Replay overlays: boxes, track id, trail, zones, stop line, signal state, trigger point.

## 20. Privacy & redaction

Redaction is a tested pipeline stage, not a UI afterthought: YuNet faces (MIT [V]) + our plate detector for non-target plates, applied at evidence-render time; originals in a restricted, role-gated, logged directory. Unit tests against synthetic overlays + a recall check on a small labeled face set [E]. Retention: raw footage deleted N days after processing (propose 30); derived evidence kept for project duration; a written deletion checklist executes after the viva. Ethics: institutional permission before recording; individual consent uncollectable in public traffic, so institutional ethics guidance governs and must start early [U].

## 21. Human-review workflow & simulated-penalty lifecycle

Reviewer actions: approve / reject / needs-more-evidence / correct-plate / annotate — all timestamped into an append-only hash-chained JSONL audit log. Penalty lifecycle: `confirmed → pending review → approved → simulated notice issued → simulated paid/contested/voided`. Every rendered artifact is watermarked **"SIMULATION — NOT A LEGAL NOTICE."** Mandatory human approval is enforced in code: an unreviewed event cannot transition to issued. The system never implies real legal enforcement capability.

## 22. Offline-first vs near-real-time demo

Primary and evaluated mode: **offline batch** (video file in, events out). **Near-real-time demo mode** runs only detector + tracker + rules on a single stream, with ANPR and evidence rendering asynchronous. This split is what the §10 compute analysis supports on the target hardware; "real-time" language is confined to the demo mode and always labeled.

## 23. Evaluation architecture

Seven protocols, evaluated separately (implemented in Phase 1+, documented in U4):
- **A. Detection** — held-out fixed-cam frames; COCO mAP@[.5:.95], AP50, per-class AP, **size-stratified AP** (small-object performance), P/R at the deployed threshold.
- **B. Helmet** — the §12 protocol.
- **C. Tracking** — HOTA/IDF1/MOTA (TrackEval, MIT [V]) on 3–5 self-annotated 30–60 s clips (semi-automated); ID switches counted **inside violation-relevant zones**.
- **D. ANPR** — the §18 protocol.
- **E. Violation events** — a predicted event matches GT iff (same rule) AND (temporal intersection ≥ 1 s OR temporal IoU ≥ 0.2) AND (median track-IoU ≥ 0.3 over the overlap, or center-containment fallback); one-to-one greedy by descending confidence; extras = false positives + duplicate-rate. Per-rule metrics: event P/R/F1; **false events per hour** (headline); median detection delay (t_confirm − t_start_GT); duplicate rate; evidence completeness (% mandatory artifacts present + hash-valid).
- **F. System** — realtime factor, per-stage latency, peak VRAM + CPU RAM, time per video-minute, crash-free hours.
- **G. Robustness** — day/night, density (weather if footage allows) slices; synthetic-corruption curves; a qualitative failure catalog with clips.

Night is a robustness **slice**, not a supported operating condition. No single "overall accuracy" is the primary claim.

## 24. Leakage-prevention policy

Frozen in U4 **before any training run**: video-level split manifest; whole-site holdout list; crop inheritance from source video; validation-only tuning (rule thresholds included — tuning on test videos is leakage too); deterministic seeded split generation; dataset checksums; repeatable evaluation commands. ~10% of event GT double-annotated for agreement statistics [R].

## 25. Storage, backend & UI responsibilities; mature repository strategy

**Storage:** SQLite (SQLAlchemy) owns events, cases, review states, audit index. Artifacts on filesystem, content-addressed (SHA-256 prefix) — manifest hash and storage layout are the same fact. Observation logs are Parquet (one file per (video, run)), the substrate for deterministic replay. Registries are YAML in-repo (`datasets.yaml`, `models.yaml`, versioned `scenes/`). Experiment tracking is file-based (config YAML in, results JSON out, git-tagged) — no tracking service.

**Backend/UI:** FastAPI serves cases, artifacts, review actions, analytics. Review UI starts as minimal server-rendered pages (HTMX-level); Streamlit is an acceptable fallback (decided by a short Phase 2 ADR, not now). Audit trail is append-only hash-chained JSONL written on every reviewer action. Nothing distributed; one machine, config-driven.

**Mature repository tree (modules appear only when a unit needs them):**
```
trafficpulse/
├── pyproject.toml
├── .github/workflows/ci.yml
├── docs/  (architecture.md, ontology.md, leakage-policy.md, evaluation-protocol.md, windows-verification.md, adr/)
├── configs/            # ontology.yaml, default thresholds, experiment configs
├── registry/           # datasets.yaml, models.yaml, splits/
├── scenes/             # versioned per-site scene configs
├── src/trafficpulse/
│   ├── contracts/                  # appears in U2
│   ├── common/                     # appears only when genuinely shared code exists
│   ├── ingestion/ · perception/ · tracking/ · association/
│   ├── observations/ · geometry/ · synth/
│   ├── scene/ · rules/ · events/
│   ├── evidence/ · anpr/ · privacy/ · calibration/
│   └── evaluation/
├── apps/  (pipeline/ , review/)
├── experiments/  (helmet_cnn_vit/, speed_gate/, anpr_study/)
├── notebooks/
└── tests/  (unit/, integration/, e2e/)
```
Reviewer test: every directory in the repo contains code exercised by at least one test. No package (`contracts/`, `common/`, or any other) is created speculatively; each appears in the unit that first implements code for it.

## 26. Ranked risk register

| # | Risk | L | I | Mitigation | Fallback | Trigger |
|---|---|---|---|---|---|---|
| 1 | Footage permissions/ethics delay | High | High | start ethics now; scout 2–3 sites | gated datasets + short recordings; narrow claims | no site by end of Phase 1 |
| 2 | AI City T5 access denied/slow | Med | High | apply now; HELMET pretraining in parallel | full custom crop annotation (~2×) | no response in 2 weeks |
| 3 | Annotation workload underestimated | High | Med | budget 35–65 h; CVAT; semi-auto MOT | cut MOT GT; shrink eval set w/ CIs | burn-down at phase boundaries |
| 4 | Integration debt (days 10–14) | Med | High | wrong-way first; contracts frozen early | ship 3–4 rules, others experimental | feature freeze at day 14 |
| 5 | Schedule risk (team capacity unknown) | High | Med | unit-level plan; parallelize U2/U3; weekly re-plan | drop optional detector comparison, night | any unit > 2× estimate |
| 6 | Indian-domain shift | Med | Med-H | IDD + own-frame fine-tune; size-stratified eval | raise min-bbox gates; near-field zones | AP-small below threshold [E] |
| 7 | Signal head not visible at site | Med | High | site criterion #1; signal-source abstraction | manual signal log for demo | site survey |
| 8 | Helmet visibility / low-res heads | High | Med | best-frame aggregation; min-crop gates; abstention | near-field-only capability | crop-size histogram |
| 9 | Rider–vehicle association errors | Med | Med | prefer crop-classifier; overlap-abstention | association path as A/B | first occupancy eval |
| 10 | Tracking ID switches | High | Med | taint logic; stationary re-association | OC-SORT swap behind contract | IDF1/switch counts |
| 11 | Indian plate OCR quality | High | Med | synthetic training; format decoding; consensus | reviewer manual transcription | exact-match on first 100 |
| 12 | Speed gate fails | Med | Med | §17 protocol; near-field; GT field day | speeding = experiment chapter | gate numbers (provisional, U4) |
| 13 | Small-object detection | Med | Med | higher input res trials; zone restriction | declare far-field out of range | size-stratified AP |
| 14 | Culturally relevant headwear | Med | Med | 4-label ontology; exempt mapping | fold to uncertain → abstain | sample counts |
| 15 | Night performance | High | Low | robustness-only framing | synthetic low-light slices | scope decision |
| 16 | Signal-state classifier unreliable | Med | Med | hysteresis; confidence gating; abstain at transitions | HSV + manual log | classifier eval |
| 17 | Windows/toolchain friction | Med | Med | no mmcv; OCR via ONNX; pinned env; Linux CI | WSL2 training only | env-setup overrun |
| 18 | Compute limits (8 GB) | Low | Med | sizes chosen for it; fp16; grad accum | smaller variants; cloud if justified | OOM in first runs |
| 19 | Dataset licensing violation | Low | High | registry gate before download | remove dataset; retrain | registry review |
| 20 | Scope creep | Med | Med | spec §26 guardrails; this review's gates | cut optional items by priority | any new capability |

## 27. Unresolved questions, ADRs, spec-revision notes, recommendation

**Unresolved [U]:** AI City T5 access; which sites secured by when (elevated view + visible signal head); ADR-001 licence posture; feasibility of a GNSS speed field day; reprocessing/event-identity semantics (ADR-004).

**Decisions requiring ADRs:**
- **ADR-001** detector/licence posture — unresolved *at review time*; may remain unresolved at the end of Phase 0-F **only** with a documented owner, deadline, and consequences; **must be resolved before detector integration** in Phase 1; does **not** block detector-independent Phase 1 work (§11). **[Resolved 2026-07-08 — Accepted: permissive-only posture, RT-DETR primary; the detector-integration gate is lifted. See [ADR-001](adr/ADR-001.md).]**
- **ADR-002** storage (SQLite + filesystem artifacts + Parquet logs) — expected Accepted.
- **ADR-003** offline-first + labeled near-real-time demo mode — expected Accepted.
- **ADR-004** reprocessing/event identity — may remain **proposed**.

Phase 2 will add a review-UI-framework ADR.

**Spec claims to revise/clarify/weaken/extend (routes; the spec file itself is not modified here):**
1. External "real-time" tagline → align to offline-first + labeled demo mode.
2. Spec §2.2 → triple riding as capability, not fixed mechanism.
3. §2.1/§4 → add exempt religious headwear to the helmet ontology (MV Act §129 [K—verify]).
4. §2.3 → add permitted movements, amber grace, encroachment-vs-junction-entry.
5. Elevate PTS/VFR discipline to ingestion-wide.
6. §6 → extend registry duties to framework/model licensing (ADR-001).
7. **New section** → evaluation-footage acquisition plan (largest omission).
8. §8 → merge dashboards 15 & 16; note 18 is file-based.
9. §12/§19 → reprocessing/event-identity as an explicit open decision (ADR-004).
10. §15 → acknowledge event-level GT annotation budget.
11. §17 → note 16 GB RAM implication for concurrent decode + dataloading.
12. State night = robustness analysis, not a supported condition.

**Recommendation: GO WITH CHANGES.** Adopt the revisions above; keep the speed gate numbers as **candidate provisional targets pending U4 justification**; keep AI City access **unconfirmed**; add the footage-acquisition plan before Phase 1; resolve ADR-001 before detector integration. No fatal flaw; no reason to revise the locked scope or the approved Phase 0-F plan.
