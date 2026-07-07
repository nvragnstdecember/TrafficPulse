# TrafficPulse Master Specification

**Document status:** Initial product and research specification\
**Purpose:** Authoritative starting point for architecture review, Phase
0 research, feasibility analysis, and phased implementation planning.

## 1. Project Thesis

TrafficPulse is an evidence-first intelligent traffic violation
detection platform for roadside video. It combines visual perception,
multi-object tracking, temporal reasoning, scene configuration and
calibration, ANPR, evidence generation, human verification, simulated
penalty issuance, and technical evaluation.

The intended high-level pipeline is:

Video Input\
→ Detection and Visual Perception\
→ Multi-Object Tracking\
→ Scene Understanding and Calibration\
→ Violation-Specific Reasoning\
→ Temporal Confidence Aggregation\
→ ANPR and Best-Frame Selection\
→ Evidence Engine\
→ Human Review\
→ Simulated Penalty\
→ Analytics and Technical Evaluation

The central principle is that TrafficPulse must not equate a single
model output or a single-frame detection with a confirmed violation. The
system should track entities over time, derive observations, accumulate
evidence, apply violation-specific temporal and geometric reasoning, and
create reviewable cases.

TrafficPulse is an academic engineering project and must optimize for: -
technical defensibility; - reproducibility; - realistic completion
time; - modularity and testability; - honest evaluation; - local
demonstration on available hardware; - clear separation between
implemented capability and future scope.

It is not intended to simulate city-scale deployment infrastructure.

## 2. Locked Violation Scope

The six supported target violations are:

1.  **No helmet**
    -   Detection of motorcycles/riders.
    -   Visual helmet-state verification.
    -   Temporal aggregation across multiple frames.
    -   Evidence selection from the most informative frames.
2.  **Triple riding**
    -   Rider detection.
    -   Rider-to-motorcycle association.
    -   Temporal rider counting.
    -   Protection against intermittent occlusion and association
        errors.
3.  **Red-light jumping**
    -   Traffic-signal state source or recognition.
    -   Vehicle tracking.
    -   Configured stop line and controlled direction.
    -   Temporal reasoning linking signal state to line crossing.
4.  **Wrong-way driving**
    -   Vehicle tracking.
    -   Trajectory history.
    -   Configured legal road/lane directions.
    -   Sustained directional contradiction rather than frame-pair
        motion.
5.  **Illegal stopping or parking**
    -   Configured restricted/no-stopping zones.
    -   Tracking and dwell-time analysis.
    -   Stationary-state reasoning.
    -   Suppression of congestion queues and legitimate red-light stops
        where relevant.
6.  **Speeding**
    -   Tracking.
    -   Scene calibration.
    -   Perspective-aware ground-plane motion estimation.
    -   Explicit uncertainty and empirical validation.

Speeding is subject to a feasibility gate. TrafficPulse must not present
monocular speed estimates as enforcement-grade measurements unless
calibration and validation justify that claim. A constrained
experimental implementation is acceptable if general speed estimation is
not defensible.

### Stretch Scope

Possible stalled-vehicle or accident-event detection is a stretch
capability only. It must not displace the six locked violations or delay
the core integrated system.

## 3. Core Architectural Principles

The following principles are mandatory unless a later architecture
review presents strong evidence for revision:

1.  **Perception and violation reasoning are separate layers.**
2.  **A model detection alone does not automatically equal a
    violation.**
3.  **Temporal evidence, track history, scene context, and explicit
    rules must be used where the violation is inherently temporal or
    geometric.**
4.  **Model outputs, derived observations, violation hypotheses,
    confirmed events, reviewed cases, and simulated penalties must
    remain conceptually distinct.**
5.  **The system is evidence-first:** every confirmed event must be
    explainable through stored evidence and rule context.
6.  **Human review is mandatory before simulated penalty issuance.**
7.  **Model-specific implementations should sit behind stable interfaces
    where practical.**
8.  **Scene geometry and policy thresholds should be versioned
    configuration data, not scattered hard-coded constants.**
9.  **Dataset provenance, licensing, split policy, leakage prevention,
    and reproducibility are first-class engineering concerns.**
10. **Privacy and redaction must be designed into the evidence workflow
    rather than added only at the UI layer.**

## 4. ViT Contribution and Research Requirement

TrafficPulse must make genuine, measurable use of Vision Transformer
technology. ViT must not be inserted into tasks merely for branding.

The initial research hypothesis is a controlled CNN-versus-ViT
comparison on an appropriate visual classification or verification task,
with helmet-state recognition as the leading candidate.

The architecture review may recommend a stronger integration point if
supported by data and feasibility evidence.

The mandatory experimental design should: - use identical
train/validation/test split policy across model families; - prevent
leakage from adjacent frames or crops from the same source video; - use
comparable preprocessing and augmentation; - use a fair and documented
training/tuning budget; - report accuracy, precision, recall, F1,
confusion matrix, latency, throughput/FPS, VRAM use, and model size; -
evaluate robustness under blur, low light, occlusion, and other relevant
conditions where data permits; - report class imbalance handling; -
include honest interpretation of negative results.

The project must not lock a specific model family before dataset and
feasibility research.

A second detector-family comparison may be considered later, but it must
not become mandatory if it threatens completion of the six-violation
integrated system.

## 5. Research-First Development Philosophy

TrafficPulse must not be developed exactly like a conventional
architecture-first software system.

The preferred workflow is:

Research\
→ Existing-material audit\
→ Dataset and licensing audit\
→ Capability decomposition\
→ Minimal baseline experiments where needed\
→ Evidence-based architecture decisions\
→ Incremental implementation\
→ Integration\
→ Layered evaluation\
→ Robustness work\
→ Report and presentation preparation

The project must avoid generating a large speculative repository
structure before validating datasets, model feasibility, and the actual
needs of the first vertical slice.

## 6. Dataset Strategy

TrafficPulse will not depend on one supposed all-in-one
traffic-violation dataset.

A dataset registry must be maintained for each learned capability. Each
dataset record should include, where applicable:

-   name;
-   official source;
-   direct source URL;
-   licence and usage restrictions;
-   version;
-   approximate size;
-   classes;
-   annotation type and format;
-   image or video modality;
-   camera perspective;
-   resolution;
-   geographic/domain characteristics;
-   Indian-road relevance;
-   day/night distribution;
-   class balance;
-   known quality limitations;
-   access status;
-   intended use;
-   train/validation/test policy;
-   provenance and verification status;
-   checksum strategy after authorized download.

Expected strategy:

### General vehicles and people

Use strong pretrained detectors first. Fine-tune only if evaluation
demonstrates that domain mismatch materially harms downstream
performance.

### Helmet state

Audit credible public motorcycle/rider/helmet datasets. Use transfer
learning and targeted augmentation. Add a small custom Indian-context
dataset only where public data is insufficient.

### Triple riding

Use public motorcycle/rider data where suitable, but expect that
rider-to-motorcycle association and event-level evaluation may require
custom video annotation.

### Red-light and wrong-way violations

Prioritize temporal reasoning and video-based event evaluation rather
than treating these as generic image-classification problems.

### ANPR

Evaluate plate detection and OCR specifically for Indian number plates,
including multi-line layouts, blur, perspective, small plate crops, and
multi-frame consensus.

### Speed estimation

Use calibrated scenes and measured ground truth. Do not treat ordinary
image labels as a substitute for speed ground truth.

### Custom data

Keep custom annotation small and purposeful. Do not spend weeks labeling
thousands of frames if pretrained models, targeted fine-tuning, and
temporal reasoning can solve the capability.

No dataset should be downloaded or used before its provenance, access
terms, and licence status are recorded and reviewed.

## 7. Capability Decomposition Requirement

For every violation, the architecture must explicitly define the chain
from input to evidence:

Raw frames\
→ required detections/classifications\
→ entity associations\
→ tracking state\
→ derived observations\
→ temporal state\
→ violation hypothesis\
→ confidence aggregation\
→ confirmation criteria\
→ evidence selection\
→ review case

Each violation design must specify: - required visual inputs; - required
perception outputs; - association requirements; - tracking
requirements; - scene configuration/calibration dependencies; - temporal
logic; - candidate/confirmation/end conditions; - confidence
components; - evidence requirements; - likely failure modes; -
abstention conditions; - feasibility for a fixed-camera capstone
demonstration.

## 8. Intended Major Technical Modules

TrafficPulse may eventually contain the following major capabilities,
but the repository should create modules only when implementation
requires them:

1.  Video ingestion and frame pipeline.
2.  Detection/perception layer.
3.  Multi-object tracker.
4.  Scene configuration and calibration.
5.  ViT visual verification component.
6.  Temporal confidence aggregation.
7.  Violation reasoning engine.
8.  ANPR pipeline.
9.  Best-evidence-frame selector.
10. Evidence package generator.
11. Event clip generator.
12. Case review workflow.
13. Simulated penalty workflow.
14. Privacy/redaction layer.
15. Traffic analytics dashboard.
16. Technical evaluation dashboard.
17. Dataset and model registry.
18. Experiment tracking and reproducibility support.

This list describes intended capabilities, not permission to scaffold
all modules immediately.

## 9. Scene Configuration and Calibration

The architecture should support versioned per-camera scene configuration
for relevant elements such as:

-   road and lane polygons;
-   legal direction vectors or angular ranges;
-   stop lines;
-   restricted/no-stopping zones;
-   parking zones;
-   traffic-light regions of interest or signal-state sources;
-   exclusion zones;
-   speed limits;
-   known-distance reference points;
-   homography or other justified perspective calibration;
-   violation-specific thresholds.

The architecture review must distinguish: - manual configuration
appropriate for a capstone demo; - automatic calibration or scene
understanding that belongs to future production work.

## 10. Temporal Reasoning and Confidence

TrafficPulse should separate: - raw detections and classifier outputs; -
tracked entities; - observations; - temporal state; - violation
hypotheses; - confidence aggregation; - confirmed violation events; -
reviewed cases.

Temporal reasoning should use explicit, testable state transitions where
appropriate. Hysteresis, minimum-duration conditions, cooldowns,
deduplication, association confidence, and abstention should be
considered.

A confidence score must not be falsely presented as a calibrated
probability unless calibration is actually demonstrated. Confidence
breakdowns should preserve component evidence such as: - detector
confidence; - classifier confidence; - temporal consistency; - track
continuity; - association ambiguity; - geometric margin; - calibration
quality.

## 11. ANPR Requirements

The ANPR feasibility study must focus on Indian number plates.

It should evaluate: - plate detection; - OCR; - Indian plate format
constraints; - multi-line layouts; - perspective correction; - blur and
low resolution; - plate-crop quality scoring; - best-frame selection; -
multi-frame OCR aggregation; - character-level confidence; - exact-match
accuracy; - privacy implications.

ANPR should be triggered or prioritized around confirmed/candidate
events rather than necessarily running expensive OCR on every vehicle in
every frame.

## 12. Evidence Engine

The Evidence Engine is a major project contribution.

A violation case should be capable of containing:

-   immutable case/event identifier;
-   violation type;
-   camera/source identifier;
-   track identity;
-   timestamps;
-   violation-specific reasoning trace;
-   relevant thresholds and measured values;
-   confidence breakdown;
-   before/trigger/after evidence frames;
-   short event clip;
-   trajectory history;
-   plate crop;
-   OCR result and confidence;
-   model versions;
-   code version where practical;
-   scene/calibration version or hash;
-   evidence artifact hashes;
-   review state;
-   audit history;
-   simulated penalty state.

A reviewer should be able to replay the event with relevant overlays
such as: - bounding boxes; - track ID; - trajectory trail; - configured
zones; - stop line; - traffic-light state; - violation trigger point.

Evidence design should support explainability and reproducibility rather
than merely producing screenshots.

## 13. Human Review and Simulated Penalty

Human review is mandatory before simulated penalty issuance.

The workflow should distinguish states such as: - detected/candidate; -
evidence packaged; - pending review; - approved; - rejected; - needs
more evidence; - simulated notice issued; - simulated
paid/contested/voided where useful for demonstration.

The system must not imply real legal enforcement capability.

Reviewer actions should be timestamped and auditable.

## 14. Privacy Requirements

The architecture must consider: - redaction of unrelated faces; -
redaction of unrelated number plates; - restricted access to unredacted
evidence; - preservation of necessary violation evidence; - retention
policy; - auditability; - responsible handling of custom-recorded
footage.

Privacy redaction should be treated as a pipeline concern with testable
behavior.

## 15. Evaluation Requirements

TrafficPulse must evaluate layers separately.

### Perception/model evaluation

Where applicable: - precision; - recall; - F1; - confusion matrix; -
per-class performance; - mAP for detection; - calibration metrics where
useful.

### Tracking evaluation

Where ground truth permits: - ID consistency; - ID switches; - IDF1; -
HOTA or other justified MOT metrics.

### ANPR evaluation

-   plate detection performance;
-   full-plate exact-match accuracy;
-   character-level accuracy/error rate;
-   OCR confidence calibration where feasible;
-   single-frame versus multi-frame consensus comparison.

### Violation-event evaluation

-   event-level precision;
-   event-level recall;
-   event-level F1;
-   false positives and false negatives;
-   false events per hour where appropriate;
-   duplicate event rate;
-   detection delay/latency;
-   evidence completeness.

### System evaluation

-   processing FPS;
-   end-to-end latency;
-   stage latency;
-   GPU memory;
-   CPU memory;
-   processing time per video minute.

### Robustness evaluation

Where data permits: - daytime; - low light/night; - blur; - occlusion; -
crowded traffic; - camera-perspective variation.

No single vague "overall accuracy" should be used as the primary claim.

## 16. Leakage Prevention and Reproducibility

Adjacent video frames are highly correlated. TrafficPulse must not
randomly split individual frames from the same sequence across train and
test sets.

The research and implementation plan must define: -
video/session/site-level split units; - whole-camera or whole-site
holdout where generalization claims require it; - inheritance of
source-video split by derived crops; - validation-only threshold
tuning; - deterministic split generation; - experiment seeds; -
model/config version tracking; - dataset provenance and checksums; -
repeatable evaluation commands.

## 17. Hardware and Environment Constraints

Primary development hardware:

-   NVIDIA RTX 4060 Laptop GPU;
-   8 GB VRAM;
-   16 GB system RAM;
-   modern CPU around 4 GHz;
-   1 TB storage;
-   Windows development environment.

The final integrated inference demonstration should run locally on this
machine.

Training may use: - mixed precision; - gradient accumulation; - smaller
batches; - image-resolution tuning; - frozen-backbone experiments; -
efficient model variants; - external/cloud GPU compute only when
genuinely justified.

Do not design around research-lab-scale models that are unrealistic for
this environment.

## 18. Speed Estimation Feasibility Gate

Speed estimation must receive special scrutiny.

The feasibility study must examine: - camera geometry; - homography or
other calibration method; - known-distance reference requirements; -
perspective distortion; - tracking error propagation; - timestamp
accuracy and variable-frame-rate concerns; - trajectory smoothing; -
measurement-zone design; - ground-truth collection; - empirical error
distribution; - uncertainty handling.

If general monocular speed estimation cannot be defended, the project
should implement a constrained calibrated-scene experiment and state its
limitations clearly.

## 19. Architecture and Storage Expectations

Only after research and feasibility analysis should the final
architecture be locked.

The architecture proposal should define: - component boundaries; - typed
data flow; - model interfaces; - scene configuration ownership; -
offline versus real-time assumptions; - backend responsibilities; -
frontend/review responsibilities; - dataset registry; - model/experiment
registry; - evidence storage; - database/index responsibilities; - audit
trail; - configuration and schema versioning.

The project should prefer capstone-appropriate storage and deployment
choices over unnecessary distributed infrastructure.

## 20. Risk Areas That Must Be Tracked

The project risk register must explicitly cover:

-   dataset availability;
-   dataset licensing;
-   domain shift to Indian urban traffic;
-   small-object detection;
-   helmet visibility and culturally relevant headwear;
-   rider-to-motorcycle association;
-   dense-traffic tracking ID switches;
-   traffic-light visibility/source reliability;
-   Indian plate OCR quality;
-   low-light/night performance;
-   speed calibration;
-   compute constraints;
-   Windows/toolchain friction;
-   integration complexity;
-   annotation workload;
-   ethics/permission delays for custom footage;
-   schedule risk;
-   scope creep.

Each risk should have mitigation, fallback, and a trigger or decision
point where practical.

## 21. Development and Milestone Philosophy

Implementation should proceed in small, independently verifiable units.

Every unit should define: - objective; - prerequisites; - expected
deliverables; - files/components likely affected; - acceptance
criteria; - tests; - required datasets; - expected compute; -
verification commands or inspection checks; - stop conditions; -
fallback approach; - dependencies.

The plan should allow parallel work only where dependencies genuinely
permit it.

An aggressive 10--14 focused-day period may target initial integration
progress, but the project must not pretend that full research-quality
experimentation, robustness evaluation, and report preparation can be
completed in that window.

## 22. Progressive Demonstration Strategy

The roadmap should support progressive demonstrations:

1.  Architecture and research review.
2.  First working perception pipeline.
3.  First end-to-end violation event.
4.  Integrated evidence workflow.
5.  Multi-violation integrated system.
6.  Final evaluated project demo.

At every milestone, implemented functionality must be clearly separated
from planned or simulated functionality.

## 23. Report and Viva Expectations

The project should generate material suitable for academic reporting,
including:

-   system architecture diagrams;
-   subsystem diagrams;
-   sequence diagrams;
-   state-machine diagrams;
-   dataset comparison tables;
-   dataset provenance tables;
-   model experiment tables;
-   confusion matrices;
-   precision/recall/F1 results;
-   latency/FPS/VRAM comparisons;
-   robustness analysis;
-   event-level evaluation graphs;
-   ablation studies where justified;
-   screenshots of evidence review;
-   failure-case analysis;
-   risk and limitation discussion.

The team should prepare for difficult viva questions about: - why ViT is
needed; - whether the CNN-vs-ViT comparison is fair; - how leakage was
prevented; - why single-frame decisions are insufficient; - how tracking
errors affect violation decisions; - how speed is calibrated and
validated; - whether ANPR results generalize to Indian plates; - how
confidence is constructed; - why the system requires human review; - how
evidence integrity is maintained; - what is actually implemented versus
simulated; - dataset licences and provenance; - failure modes and
limitations.

## 24. Initial Phase 0 Objective

Phase 0 is research, audit, feasibility analysis, and planning.

Before production implementation, Phase 0 should: - audit all existing
TrafficPulse material; - decompose every violation capability; -
research and verify dataset candidates; - identify dataset gaps and
custom-data needs; - compare realistic model/method options; - design
the mandatory ViT experiment; - design temporal reasoning boundaries; -
design scene configuration/calibration requirements; - evaluate speed
feasibility; - evaluate Indian ANPR feasibility; - design evidence and
privacy workflows; - define layered evaluation; - propose a technically
defensible architecture; - create a ranked risk register; - produce a
phased implementation roadmap; - map milestones to presentation and viva
needs.

Phase 0 must not: - implement production code; - scaffold a speculative
application architecture; - install project dependencies; - download
datasets; - train models; - make destructive changes; - claim
verification that did not occur.

## 25. Definition of Success

TrafficPulse succeeds if it demonstrates a credible end-to-end evidence
workflow for the locked violations with honest evaluation and clear
limitations.

A successful final system should be able to show, for supported
scenarios:

Video\
→ perception\
→ tracking\
→ scene-aware temporal reasoning\
→ violation event\
→ evidence package\
→ ANPR where feasible\
→ privacy-aware human review\
→ simulated penalty workflow\
→ analytics and technical evaluation

The project should be judged not by maximum feature count, but by: -
correctness of reasoning; - quality of evidence; - experimental rigor; -
reproducibility; - honest handling of uncertainty; - system
integration; - academic defensibility.

## 26. Non-Negotiable Scope Guardrails

-   The six locked violations remain the core target.
-   Accident/stalled-vehicle detection is stretch-only.
-   ViT must have a genuine evaluated role.
-   No model is selected purely because it is fashionable.
-   No dataset is trusted without provenance and licence review.
-   No random frame-level train/test leakage.
-   No single-frame violation decision where temporal evidence is
    required.
-   No enforcement-grade speed claim without calibration and validation.
-   No simulated penalty without human review.
-   No production-scale distributed architecture unless evidence proves
    it necessary.
-   No large speculative scaffold before research and architecture
    decisions justify it.
-   No claim that a command, test, dataset, model, or integration was
    verified unless it was actually inspected or run.
