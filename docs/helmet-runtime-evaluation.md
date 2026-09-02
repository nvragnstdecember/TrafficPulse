# Helmet classification on the runtime path — P4-U8, P4-U9, P4-U10

- **Status:** complete; **no production backend is adopted by this document**
- **Date:** 2026-09-01
- **Protocols:** [`PROTOCOL_P4U8.md`](../experiments/helmet_runtime_validation/PROTOCOL_P4U8.md),
  [`PROTOCOL_P4U9.md`](../experiments/helmet_runtime_validation/PROTOCOL_P4U9.md),
  and P4-U6-V's [`PROTOCOL.md`](../experiments/helmet_runtime_validation/PROTOCOL.md)
- **Relationship to P4-U5 / ADR-005:** P4-U5 (`p4u5-prereg`, `52839d0`) is a **read-only
  input**. Nothing here amends it, its results, or ADR-005. ADR-005 decision 3 adopts no
  backend and decision 4 gates adoption on evaluating candidates *on derived head crops*;
  this document is that evaluation, and it does not by itself lift the gate.

---

## 1. Summary in one paragraph

Correcting a mistake in the **evaluation scaffold** — not in the detector — raises measured
runtime recovery on the frozen HELMET test split from **53.7% to 74.4%**. Classifier
accuracy on the recovered crops is essentially unchanged by the correction (ResNet forced
macro-F1 0.8161 -> 0.8142), which is the expected result and a good sign: the correction
moved the *denominator*, not the models. On the corrected population ResNet-50 leads
DeiT-Small by **+0.0337 macro-F1** with a bootstrap CI of **[+0.0155, +0.0522]** and
McNemar **p = 0.0084**; both trained models beat the production zero-shot backend by roughly
**+0.50 macro-F1**, which is not a close call. Despite that, **no backend is declared the
production winner**, because the trained models cannot emit `turban`, the evaluation rests
on one checkpoint per family, and real-video inspection shows per-frame label instability
that the violation rule has never been tested against.

## 2. P4-U8 — runtime recovery re-measurement

### 2.1 The correction

A HELMET annotation row is one box per tracked motorcycle and that box encloses **the
motorcycle together with its riders**. RT-DETR's `motorbike` box encloses the **vehicle
only**. P4-U6-V matched those two boxes by IoU, which compares different objects.

Measured on **val** before the rule was frozen (geometry only, no recovery rate involved):
in **99.95%** of overlapping cases the vehicle box starts *below* the annotation's top edge,
by a median of **36.5% of the annotation's height**, and the vehicle box is a median
**63%** of the annotation's height. That is the rider's head and torso, missing.

The corrected rule (PROTOCOL_P4U8 §3) matches the annotation against a **rider-inclusive
evaluation proxy**: the union of the runtime motorcycle box with the boxes of the riders the
production `associate_riders` linked to it, at the **unchanged** IoU floor of 0.50.

**This is an evaluation-scaffold correction, not a detector improvement.** The detector,
its checkpoint, its 640x640 non-uniform input, its 0.50 score threshold, the tracker, the
association policy and the head-crop geometry are all identical to P4-U6-V's. No production
file was changed by this unit.

### 2.2 The reconstruction is proven, not assumed

The analysis runs offline from P4-U7's detection record. Its fidelity is demonstrated
rather than argued:

- run in `motorcycle_only` mode it reproduces P4-U6-V's recovered crop-id set **exactly** —
  3,528 / 3,528 ids and all four per-reason counts identical;
- the 3,508 crops that survive into the corrected population are **byte-identical PNGs** to
  P4-U6-V's;
- re-scoring those identical crops reproduces the same decisions: max probability
  difference 3.7e-6 (float32 batching noise) and **zero** arg-max flips across all three
  backends.

### 2.3 Recovery, before and after

| | val | test |
|---|---|---|
| total frozen crops | 4,089 | 7,406 |
| single-rider eligible | 2,284 (55.9%) | 4,269 (57.6%) |
| multi-rider excluded | 1,805 (44.1%) | 3,137 (42.4%) |
| **recovered — P4-U6-V convention** | 1,236 (54.1%) | 2,292 (53.7%) |
| **recovered — corrected convention** | **1,704 (74.6%)** | **3,175 (74.4%)** |

Test taxonomy (P4-U7 buckets), motorcycle-only -> rider-inclusive:

| bucket | before | after |
|---|---|---|
| A — nothing overlapping | 403 | 310 |
| B — detected but below the IoU floor | 1,360 | 556 |
| C — matched but association failed | 214 | 228 |
| D — recovered | 2,292 | 3,175 |

Detection-failure rate (A+B, of eligible) falls **41.3% -> 20.3%**. Association failure is
**not** repaired by the correction: it stays at ~5% of eligible (5.01% -> 5.34%) and rises
in absolute count, because more motorcycles now match at all. As a share of *matched*
motorcycles it falls 8.5% -> 6.7%.

### 2.4 The correction is not buying matches

| diagnostic (test) | value |
|---|---|
| median IoU of the matched proxy | 0.908 (was 0.606) |
| median *motorcycle-only* IoU of the same matches | 0.570 |
| suspicious matches (motorcycle-only IoU < 0.10) | 14 / 3,403 = **0.41%** |
| contested proxies (>1 annotation clears 0.50) | 29 |
| crops recovered before but not after | 20 (0.57%) |

The matched proxies fit the annotations far better *and* their underlying vehicle boxes
already overlapped the annotation substantially. Only 14 matches were carried by the rider
union alone.

### 2.5 Residual issues, re-measured under the corrected convention

Every problem P4-U7 identified is still there; the correction shrinks but does not remove
them.

**By annotation-area quartile (test):**

| quartile | eligible | before | after |
|---|---|---|---|
| Q1 (smallest) | 1,068 | 23.2% | **59.6%** |
| Q2 | 1,068 | 54.3% | 75.7% |
| Q3 | 1,066 | 64.9% | 78.6% |
| Q4 (largest) | 1,067 | 72.4% | 83.5% |

Small objects remain the worst stratum by a wide margin.

**By site (test), worst five after correction:** Pakokku_urban 51.2% (from 15.9%; still
159 of its 510 eligible riders have **no** overlapping detection at all), NyaungU_urban
64.0%, Yangon_II 66.2%, Mandalay_2 72.7%, Bago_urban 75.6%. Best: NyaungU_rural 94.9%,
Naypyitaw_1 92.5%, Pathein_rural 90.6%.

**By class (test):** helmet 76.7%, no_helmet **67.9%**. The class-biased recovery narrows
(gap 10.7pp -> 8.7pp) but persists, so the recovered population still under-represents the
class the violation rule cares about (24.0% recovered vs 26.3% eligible).

**By rider count (test):** rider counts 1 / 2 / 3 / 4 / 5 = 4,269 / 2,671 / 406 / 48 / 12.
Only rider-count 1 is evaluated; the rest are reported as `not_evaluated`, never as zero.

## 3. P4-U9 — classifier re-evaluation on the corrected population

Population: 1,704 val and 3,175 test crops, single-rider only, all three backends scoring
the **same** PNG files. Thresholds selected on val only; the selection returned exactly
P4-U6-V's values (zero-shot `None`, ResNet 0.80, DeiT 0.60), so the pre-declared
sensitivity check is identical to the primary result — the operating points are stable
under the corrected population.

### 3.1 Test results

**A. Forced binary choice (primary; coverage 100% by construction):**

| backend | macro-F1 | accuracy | ECE | helmet P/R/F1 | no_helmet P/R/F1 |
|---|---|---|---|---|---|
| Zero-shot | **0.2862** | 0.3071 | 0.518 | 0.982 / 0.090 / 0.164 | 0.257 / 0.995 / 0.408 |
| ResNet-50 | **0.8142** | 0.8564 | 0.090 | 0.930 / 0.877 / 0.903 | 0.671 / 0.790 / 0.726 |
| DeiT-Small | **0.7805** | 0.8365 | **0.026** | 0.900 / 0.883 / 0.891 | 0.651 / 0.689 / 0.670 |

**B. Native operating point (deployment-shaped):**

| backend | abstain_below | macro-F1 | accuracy | coverage | abstentions | ECE |
|---|---|---|---|---|---|---|
| Zero-shot | — | 0.2973 | 0.3219 | 93.0% | 221 (all `turban`) | 0.411 |
| ResNet-50 | 0.80 | **0.8567** | 0.8936 | 89.7% | 328 | 0.085 |
| DeiT-Small | 0.60 | 0.8184 | 0.8716 | 91.0% | 285 | **0.022** |

Note ResNet's test coverage of **89.7%** slips just under the 90% floor the val selection
guaranteed. That is a val/test generalisation gap in the operating point, not a violation
of the protocol, and it is recorded rather than re-selected.

Confusion matrices (forced view, `[true][predicted]`):

- zero-shot — helmet: 216 / 2,196; no_helmet: 4 / 759
- ResNet — helmet: 2,116 / 296; no_helmet: 160 / 603
- DeiT — helmet: 2,130 / 282; no_helmet: 237 / 526

**The zero-shot backend is not a working helmet classifier on runtime crops.** It calls
`no_helmet` on 93% of everything, recovering 99.5% of true `no_helmet` at 25.7% precision.
Its macro-F1 of 0.286 is *below* what a constant predictor of the majority class would be
judged worth, and its ECE of 0.52 means its scores carry almost no calibration information.

### 3.2 Pairwise comparisons (forced view, all 3,175 crops)

| comparison | Δ macro-F1 | 95% bootstrap CI | McNemar p | discordant |
|---|---|---|---|---|
| ResNet vs DeiT | **+0.0337** | [+0.0155, +0.0522] | **0.0084** | 555 (309 / 246) |
| Zero-shot vs ResNet | −0.5280 | [−0.5476, −0.5084] | < 1e-300 | 2,118 |
| Zero-shot vs DeiT | −0.4943 | [−0.5147, −0.4736] | < 1e-300 | 2,177 |

Against P4-U6-V the ResNet-vs-DeiT point estimate is stable (+0.0313 -> +0.0337) but the
larger population tightens the interval and moves McNemar from p = 0.157 to p = 0.0084.
The advantage is now statistically detectable. It is still **+3.4 macro-F1 points from one
checkpoint per family**, which is not the same thing as a production winner (§5).

McNemar is computed with the log-space implementation written for P4-U6-V. The frozen
`helmet_cnn_vit.stats.mcnemar` overflows above ~1,023 discordant pairs and is **not
patched**, because it belongs to a tagged published experiment.

### 3.3 Where the difference comes from

| stratum | crops | zero-shot | ResNet | DeiT |
|---|---|---|---|---|
| shared with P4-U6-V | 2,280 | 0.2823 | 0.8172 | 0.7850 |
| **added by the correction** | 895 | 0.2932 | 0.8062 | 0.7673 |
| head region < 30px | 306 | 0.1934 | **0.7112** | **0.6112** |
| head region >= 30px | 2,869 | 0.2957 | 0.8247 | 0.7995 |

The crops the correction added are only marginally harder, which is further evidence the
rule did not admit junk. The **small-crop stratum is where both trained models fall apart**:
ResNet loses 11 macro-F1 points and DeiT loses 19 below 30px. DeiT degrades on small crops
roughly twice as fast as ResNet — the clearest practical difference between them.

### 3.4 End-to-end honesty

Conditional accuracy is not system performance. Per **annotated motorcycle** in the frozen
test split:

```
0.576 eligible (single-rider)  x  0.744 recovered  x  0.897 classifier coverage
    = 0.384 -> ResNet produces a helmet decision for ~38% of annotated motorcycles
```

Before the correction the same figure would have been ~0.31. Neither is a working
end-to-end system, and the 42.4% multi-rider exclusion is the single largest term.

## 4. Turban, and multi-rider

### 4.1 Turban

`turban` is never mapped to `no_helmet` and the capability guard
(`classifier.capabilities.require_turban_capability`) is never bypassed. HELMET carries no
turban annotation, so no turban prediction can be scored right or wrong here.

On the corrected test population the zero-shot backend — the **only** production backend
that can emit the label — predicts `turban` on **221 / 3,175 crops (6.96%)**. That is its
entire abstention: it has no `uncertain` prompt by design, so `uncertain` is structurally
impossible for it, and turban abstentions alone cost it 7 points of coverage.

Of those 221 turban predictions, **214 (96.8%) are on riders the annotation records as
wearing a helmet** and 7 on riders recorded bare-headed. On real video the pattern repeats
in a place where turbans are close to absent from traffic: **20 / 143 crops (14%) in the
Thailand clip** are called `turban`.

Read together, the most likely explanation is that CLIP's turban prompt is firing largely on
**helmeted** riders rather than detecting turbans. That is a hypothesis, not a measurement —
proving it needs turban ground truth this project does not have — but it means the turban
capability the zero-shot backend nominally provides is **not demonstrably a working turban
detector**, and the exemption it feeds is therefore not demonstrably working either.

**Why the binary models cannot replace it.** ResNet-50 and DeiT-Small have a two-class head.
No configuration makes them emit `turban`; `supported_labels` declares this, and
`require_turban_capability` refuses to build a no-helmet rule around them unless an operator
explicitly records the consequence. Adopting either as the production backend today would
mean either (a) running the no-helmet rule turban-blind, in which case a turban-wearing
rider is confirmed as a violation — a systematic false-positive class against a religious
group — or (b) acknowledging the guard, which records the choice but does not fix it. This
document does **not** choose a turban architecture.

### 4.2 Multi-rider

`observations.helmet.rider_slot` returns `DRIVER` only when exactly one rider is associated
and `UNKNOWN` otherwise, because the shipped `IouTracker` supplies no velocity and
driver-versus-pillion cannot be read off image-space position. No driver/pillion heuristic
is invented anywhere in this work.

Consequently **42.4% of the frozen test corpus (3,137 of 7,406 annotated motorcycles) is
excluded from every quantitative claim in this document**, with rider counts of 2 (2,671),
3 (406), 4 (48) and 5 (12). Nothing measured on single-rider crops is evidence about
multi-rider traffic, and in the real footage audited below multi-rider is not an edge case:
**157 of 194 crops (81%)** in the Bihar congestion clip sat on a motorcycle the system
believed carried 2-4 riders.

## 5. P4-U10 — real video, qualitatively

Three clips through the production runtime path, tracker running continuously over
consecutive frames, every frame rendered with its boxes, associations and predictions so
the findings could be *seen* rather than inferred. No threshold, label or model was changed
on the basis of anything observed. The no-helmet violation rule was **not** run: doing so
would require acknowledging the turban guard for a binary backend, which is the bypass the
guard exists to prevent.

| clip | frames | crops | multi-rider crops | crops < 30px head | frames with no motorcycle |
|---|---|---|---|---|---|
| Gangtok congestion (India, handheld portrait) | 60 | 36 | 0 | 0 | 30 |
| Raxaul congestion (India, handheld, dense crowd) | 60 | 194 | 157 (81%) | 0 | 4 |
| Chiang Mai intersection (Thailand, elevated static) | 60 | 143 | 12 | 86 (60%) | 20 |
| Contraflow roundabout (Australia) | 60 | 0 | — | — | 60 |

### 5.1 Findings by category

1. **Helmet / no helmet.** On the elevated Thailand clip the runtime produces plausible
   per-rider calls with visible helmets. The three backends agree on only **25 / 143** crops
   there, **4 / 36** on the Gangtok clip and **95 / 194** on the Bihar clip. ResNet and DeiT
   agree far more with each other (68%, 64%, 82%) than either does with zero-shot.
2. **Multiple riders.** Handled honestly and uselessly: `rider_slot` returns `unknown`, so
   no driver attribution exists. On the Bihar clip one motorcycle was linked to **three
   "riders" who are visibly separate people in a level-crossing crowd** — the documented
   "overlap is not riding" limitation of the IoMin association, seen operating.
3. **Small / distant riders.** The Thailand clip's median head region is **28.4px**
   (min 12.5px, Q1 21.7px): 60% of its crops fall in the <30px stratum where §3.3 measures
   ResNet at 0.711 and DeiT at 0.611 macro-F1. A realistic elevated-CCTV view puts most
   riders in the *worst* measured regime.
4. **Difficult lighting.** Not isolated: no night or rain clip is downloaded in
   `test-videos/`. This category is **untested**, and is stated as untested rather than
   passed.
5. **Occlusion.** Visible and damaging on both congestion clips, where foreground vehicles
   and crowds cut riders in half. In the Bihar clip the full-width head geometry produces
   crops with a **median width of 714px** — a band across a crowd, in which the head is a
   small minority of the pixels. This is the documented containment-over-tightness cost, and
   at close range it degenerates.
6. **Turban / head coverings.** See §4.1: 14% of Thailand crops called `turban` in a
   population where turbans are close to absent.
7. **Temporal instability (not on the original list, but the most consequential finding).**
   ResNet's arg-max label **flips between consecutive frames on 7 of 11 tracks** in the
   Thailand clip, 2 of 3 in Gangtok and 4 of 16 in Bihar; DeiT flips on 8 of 11, 2 of 3 and
   7 of 16. On one Gangtok rider ResNet alternates helmet 0.99 / no_helmet 0.78 / helmet
   0.99 across successive frames. The violation rule depends on sustained temporal runs, and
   **it has never been evaluated against classifier output this unstable**.

### 5.2 Correct negatives worth recording

The Australian contraflow clip contains no motorcycles; the system produced no crops, which
is right. Window 2 of the Gangtok clip likewise genuinely contains none. Pedestrians
standing near vehicles were not associated as riders in either.

## 6. Demo-readiness audit

The distinction applied throughout: *technically runs* / *scientifically defensible* /
*safe to claim*.

| # | Demo | Verdict | Why |
|---|---|---|---|
| A | Detection-only | **READY** | RT-DETR at 0.50 runs on all four clips; boxes are visibly correct on elevated views. Disclose small-object recall: Pakokku_urban still has 31% of eligible riders with no overlapping detection. |
| B | Runtime classifier (crop -> label) | **READY WITH LIMITATIONS** | Numbers are real, pre-registered and reproducible. Must be presented as *conditional* accuracy on single-rider crops, with the 74.4% recovery and 42.4% multi-rider exclusion stated in the same breath. Never quote P4-U5's 0.929. |
| C | Full helmet violation (rule fires) | **NOT READY** | Requires either a turban-blind binary backend (guard bypass, systematic false positives against turban wearers) or the zero-shot backend, whose runtime macro-F1 is 0.286. The temporal rule has never been tested against the per-frame instability of §5.1(7). |
| D | Multi-rider | **NOT READY** | `rider_slot` is `UNKNOWN` for every multi-rider motorcycle by design, so there is no driver attribution to demo. 81% of crops in a real congestion clip are in this state, and association links non-riders in crowds. |
| E | Turban / exemption | **NOT READY** | Only the zero-shot backend can emit `turban`; 96.8% of its turban calls on the test split land on riders annotated as helmeted. Demonstrating an exemption on that evidence would be demonstrating an artefact. |
| F | End-to-end video upload | **READY WITH LIMITATIONS** | The upload -> process -> review workspace works and is shippable for detection and for the other violation families. The helmet violation must be **off** or clearly labelled experimental in any such demo, per C. |

"Technically runs" covers A, B, D, F. "Scientifically defensible to present" covers A and B
with their disclosures, and F for non-helmet violations. "Safe to claim in a viva" covers A,
B-with-disclosure, and F-with-disclosure — and explicitly **not** C, D or E.

## 7. Production adoption decision

| Backend | Runtime performance | Calibration | Abstention | Turban | Deployment | Statistical confidence | Recommendation |
|---|---|---|---|---|---|---|---|
| **Zero-shot (CLIP)** | macro-F1 **0.286**; predicts `no_helmet` on 93% of crops | ECE 0.52 (forced) / 0.41 (native) — unusable | 7.0%, entirely `turban`; no `uncertain` prompt exists | Only backend that emits it — but 96.8% of its turban calls are on helmeted riders | Shipped, production seam, is the current default | Beaten by both trained models by ~0.50 macro-F1, p < 1e-300 | **Not fit for the violation rule.** Its only unique capability is not demonstrably working. |
| **ResNet-50** | macro-F1 **0.814** forced / **0.857** native; best in every stratum | ECE 0.090 — usable but over-confident | 10.3% at 0.80; test coverage 89.7%, just under the 90% floor | **Cannot emit it** | Production wrapper exists (`classifier/resnet.py`) | +0.034 over DeiT, CI [+0.016, +0.052], p = 0.008; one seed | **Best measured candidate; adoption still blocked by turban.** |
| **DeiT-Small** | macro-F1 **0.781** forced / **0.818** native; degrades ~2x faster than ResNet below 30px | **ECE 0.026 — much the best calibrated** | 9.0% at 0.60 | **Cannot emit it** | **No production wrapper**; the checkpoint is a `timm` ViT torchvision cannot load, scored through the research env | Behind ResNet, but its calibration edge is large and orthogonal | **Keep as the ViT arm of the record; not deployable today.** |

**Recommendation: fix another system component first — specifically the turban capability —
and retain the current zero-shot default behind the existing guard in the meantime.**

Reasoning, stated plainly:

- ResNet is clearly the strongest classifier on the runtime crops, and if the only question
  were "which model reads a crop better", it would be the answer.
- But adopting it would trade a backend that is **bad at the primary task** for one that is
  **incapable of the exemption**, and the exemption's failure mode is a systematic false
  accusation against a protected group. That is not an improvement, it is a different and
  worse failure.
- The zero-shot backend should not be defended either: at macro-F1 0.286 it is not doing the
  job, and its turban capability is not demonstrably real. The honest position is that
  **neither option is safe today**, which is why no winner is declared.
- A one-seed comparison cannot separate an architecture effect from a checkpoint effect.
  ResNet-vs-DeiT is +0.034 with CI [+0.016, +0.052] from **one checkpoint per family**;
  P4-U5's three-seed rule is deliberately not imported (PROTOCOL_P4U9 §6.3).

The unblocking options, in rough order of cost, are: compose a binary backend with a
turban-capable one behind the existing capability declaration; derive turban on its own
evidence path; or retrain with a turban class on data that has one. Choosing between them
is a project decision, not a side effect of this evaluation.

## 8. Completion assessment

| # | Question | Verdict | Basis |
|---|---|---|---|
| 1 | Is TrafficPulse technically runnable? | **GREEN** | Full path runs on four real clips and the frozen corpus; upload -> process -> review workspace works. |
| 2 | Is it demoable? | **YELLOW** | Yes for detection and for non-helmet violations; the helmet violation must be off or labelled experimental. |
| 3 | Is the helmet classifier production-ready? | **RED** | Best candidate is capable (0.857 native macro-F1) but cannot emit `turban`; the shipped default scores 0.297. |
| 4 | Is the multi-rider path production-ready? | **RED** | No driver attribution exists by design; 42.4% of the corpus and 81% of a real congestion clip are unattributable. |
| 5 | Is turban/exemption handling production-ready? | **RED** | The guard is correct and working; the *capability* behind it is not demonstrably real. |
| 6 | Is the detector/evaluation pipeline trustworthy? | **GREEN** (evaluation) / **YELLOW** (detector) | The scaffold is now correct and its fidelity is proven by exact reproduction. The detector still misses ~20% of eligible riders and 40% of the smallest quartile. |
| 7 | What remains before a defensible viva? | see §9 | |
| 8 | What remains before calling the project complete? | see §9 | |

## 9. Remaining blockers and next steps

**Blockers (must be fixed before claiming the helmet feature is complete):**

1. **Turban capability.** No safe backend choice exists until this is resolved. *(RED)*
2. **Driver/pillion attribution.** Needs a motion-capable tracker (ByteTrack/OC-SORT) or
   temporal heading derivation; without it 42% of traffic is unattributable. *(RED)*
3. **Temporal stability of classifier output.** Per-frame labels flip on the majority of
   real tracks; the no-helmet rule's run semantics have never been evaluated against that.
   *(RED)*

**Disclosures (safe to demo provided they are stated):**

4. Recovery is 74.4%, not 100%: ~20% of eligible riders never reach the classifier. *(YELLOW)*
5. Small-object performance: 40% of the smallest quartile is still unrecovered, and ResNet
   drops to 0.711 macro-F1 below 30px. *(YELLOW)*
6. Class-biased recovery: `no_helmet` recovers 8.7pp below `helmet`. *(YELLOW)*
7. The head-crop geometry degenerates at close range (median 714px-wide crops on one clip).
   *(YELLOW)*
8. Difficult-lighting behaviour is **untested** — no night or rain clip is available. *(YELLOW)*
9. One checkpoint per family; no seed-consistency claim is available for this comparison.
   *(YELLOW)*
10. The P4-U5 JPEG q95 training round-trip is still not reproduced at runtime; a residual
    domain gap remains, by choice. *(YELLOW)*

**Next steps, in order:** decide the turban architecture; evaluate the no-helmet rule
end-to-end against real per-frame classifier output (with temporal aggregation) before any
adoption; then, and only then, revisit backend adoption with a multi-seed criterion.
