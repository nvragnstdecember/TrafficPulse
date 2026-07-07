# TrafficPulse Scene Configuration

**Version:** 1.0.0 · **Status:** frozen (Phase 0-F, U5)
**Schema:** [`configs/scenes/schema.yaml`](../configs/scenes/schema.yaml) ·
**Example:** [`configs/scenes/example-scene.yaml`](../configs/scenes/example-scene.yaml)
**Architecture ref:** `docs/architecture-review.md` §16 · **Consumes U2/U3/U4** for closed sets and policy.

Per-camera scene configuration is **declarative data**. U5 freezes the YAML
schema, a frozen coordinate convention, closed vocabularies, a synthetic example,
a **typed Pydantic contract** (`src/trafficpulse/contracts/scene.py`) with a
**deterministic `scene_config_hash`**, and focused validation tests. **U5
implements no behaviour** — no point-in-polygon, line crossing, heading
comparison, homography maths, signal classification, rule engine, or calibration.
Those are Phase 1+ and consume this data.

The YAML file is the human-editable representation; `SceneConfig` is the typed
validation + hashing contract that Phase 1 loads it into.

## 1. Purpose and status
Freeze the scene-configuration contract before any geometry/rule/calibration
code exists, so Phase 1 builds against a stable, validated data shape. The
example is synthetic and for schema validation/demo only.

## 2. File structure
`configs/scenes/schema.yaml` (required sections/fields, vocabularies, reference
categories, security rules) and one or more scene files such as
`configs/scenes/example-scene.yaml`. Sections: `scene`, `frame`, `zones`,
`stop_lines`, `legal_directions`, `signal_groups`, `speed_limits`,
`calibration`, `rule_parameters`.

## 3. Coordinate convention (frozen)
- **Space:** pixel (not normalized).
- **Origin:** top-left `(0,0)`.
- **Axes:** `+x` right, `+y` down.
- **Polygons:** an ordered ring of `[x, y]` vertices; the ring is closed
  implicitly (do not repeat the first point).
This matches the U2 `BoundingBox` convention. All image-space points must lie
within `[0, reference_width] × [0, reference_height]`.

## 4. Zone semantics
Named polygonal regions with a `zone_type` from a closed set (`lane`, `approach`,
`exit`, `intersection`, `no_stopping`, `speed_measurement`,
`signal_controlled_region`, `roi`). A zone may declare `applicable_violations`
(U2 `ViolationType` ids), `observation_consumers` (U2 observation discriminators),
an optional `legal_direction_id`, and an optional `signal_group_id`. Zones carry
**no** temporal logic.

## 5. Stop lines
A segment (`endpoints.a`, `endpoints.b`) with a `crossing_direction` vector
describing the intended legal crossing orientation, plus references to a
`signal_group_id` and `zone_ids`. The data lets later code decide crossing
orientation **without** any segment-intersection algorithm being defined here.

## 6. Legal directions
A `direction_id`, a non-zero `vector` (`{dx, dy}`), associated `zone_ids`, and a
description. Vectors are **not** normalized and headings are **not** compared in
U5; a zero-length vector is rejected by validation. An optional
`tolerance_degrees` may be provided only as provisional configuration (the
example leaves it unset — the deviation threshold lives in `rule_parameters`).

## 7. Signal groups and ROIs
A `signal_group_id`, an `roi` (rectangle or polygon within frame bounds), a
`signal_source_mode` (`roi_classifier`, `simulated_schedule`, or
`manual_annotation`), controlled `stop_line_ids`/`zone_ids`, and
`expected_states` (U2 `SignalState` ids). If `simulated_schedule` is ever used it
is **demo/testing configuration only** — no live controller integration exists.
The example uses `roi_classifier`.

## 8. Speed-limit context
A `speed_limit_id`, an explicit `value` + `unit` (`km_per_h` or `m_per_s`;
km/h for this project), associated `zone_ids`, a `source`, and a
`verification_status`. Limits are **not** inferred; the example uses a clearly
synthetic demo value marked `unverified`.

## 9. Calibration metadata
Declarative metadata sufficient for future speed estimation: `calibration_id`,
`type`, `status`, `verification_status`, `source`, `created_at`, `world_unit`, a
`homography_matrix` (3×3 config data), `correspondences`, `quality_metrics`, and
`notes`. **U5 computes nothing** — no homography solve, no inversion, no point
transform, no reprojection error, no accuracy claim. The example's matrix is a
synthetic identity placeholder; `status: provisional`,
`verification_status: unverified`, `reprojection_rmse_px: null`.

## 10. Provisional rule parameters
Per-violation configuration blocks of `{id, value, unit, status, note}` items.
`status` is `unset` (with `value: null`) or `provisional` — **never** `tuned` or
`validated` in this unit. Provisional defaults carried from the architecture
review (e.g. wrong-way deviation ~120°, persistence ~1.0 s, min speed ~1.5 m/s;
illegal-stop ~10 s, motion ~0.5 m/s; red-light grace ~0.3 s) are **provisional
only** and require tuning/validation on held-out data. Where no value is
justified, the parameter is `unset` rather than inventing a threshold. These are
**configuration values, not behaviour**: no thresholds are applied in U5.

## 11. Provenance and verification statuses
`scene.status` ∈ {draft, calibration_pending, validation_pending, validated,
archived}; `calibration.status` ∈ {absent, provisional, unverified, validated,
rejected}; `verification_status` ∈ {unverified, project_verified,
externally_verified}; `parameter_status` ∈ {unset, provisional, tuned,
validated}. The example is `draft`, its calibration `provisional`/`unverified`,
and every parameter `provisional`/`unset` — nothing is over-claimed.

## 12. Security and privacy rules
Scene files must **never** contain RTSP URLs, credentials, usernames, passwords,
API keys, tokens, private keys, connection strings, precise addresses, or
geolocation. Camera and site references are **opaque identifiers**. Tests sweep
for forbidden key names and credential-like value patterns.

## 13. Change control
Scene config is versioned (`config_version`, `schema_version`). Any change that
affects evaluation must be versioned, and — per
[`docs/dataset-policy.md`](dataset-policy.md) — **held-out evaluation footage
must not be used to tune scene parameters** (validation-only tuning). Changing
the schema's closed sets is a reviewed schema-version change.

## 14. Typed contract, hashing, and Phase 1 relationship
U5 provides the typed `SceneConfig` contract and a deterministic
`scene_config_hash` (stdlib SHA-256). The **canonical serialization boundary**
for hashing is the model's JSON-mode dump serialized with
`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)` and
UTF-8 encoded: the model normalizes structure (field order, enum values,
int→float, datetimes→ISO) and `sort_keys` normalizes mapping key order, so
semantically identical configurations hash identically regardless of YAML
formatting or key order. The 64-character lowercase hex digest is compatible with
the `Sha256Hex` `scene_config_hash` field on `ConfirmedEvent` /
`EvidenceManifest`.

Phase 1 will implement the geometry, calibration, signal, and rule **behaviour**
that consume this data. None of that behaviour exists in U5 — this unit only
freezes the declarative contract, its validation, and its stable hash.
