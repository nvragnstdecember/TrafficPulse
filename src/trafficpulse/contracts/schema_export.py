"""Deterministic JSON Schema export for the public U2 contracts.

Design goals:
- byte-identical output across runs given identical source;
- inspectable, diff-friendly files suitable for future interface validation;
- minimal — no code-generation framework.

Determinism is achieved by serializing each schema with sorted keys, fixed
2-space indentation, UTF-8, and a trailing LF, then writing raw bytes so no
platform newline translation can perturb the output.

Run as a script to (re)generate ``schemas/``::

    python -m trafficpulse.contracts.schema_export
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .association import Association
from .detection import Detection
from .event import ConfirmedEvent
from .evidence import EvidenceManifest
from .hypothesis import ViolationHypothesis
from .observations import OBSERVATION_VARIANTS, ObservationAdapter
from .penalty import SimulatedPenalty
from .review import ReviewCase
from .temporal import TemporalState
from .track import TrackState

# The nine top-level contract models exported individually.
TOP_LEVEL_CONTRACTS: tuple[type[BaseModel], ...] = (
    Detection,
    TrackState,
    Association,
    TemporalState,
    ViolationHypothesis,
    ConfirmedEvent,
    EvidenceManifest,
    ReviewCase,
    SimulatedPenalty,
)

# Filename used for the discriminated Observation union schema.
OBSERVATION_UNION_NAME = "Observation"

DEFAULT_OUTPUT_DIR = Path("schemas")


def _serialize(schema: dict[str, Any]) -> bytes:
    text = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def export_schemas(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Write all public contract schemas into ``output_dir``.

    Exports: the nine top-level contracts, each of the seven observation
    variants, and the discriminated Observation union. Returns the sorted list
    of written paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    models: tuple[type[BaseModel], ...] = TOP_LEVEL_CONTRACTS + OBSERVATION_VARIANTS
    for model in models:
        path = output_dir / f"{model.__name__}.schema.json"
        path.write_bytes(_serialize(model.model_json_schema()))
        written.append(path)

    union_path = output_dir / f"{OBSERVATION_UNION_NAME}.schema.json"
    union_path.write_bytes(_serialize(ObservationAdapter.json_schema()))
    written.append(union_path)

    return sorted(written)


def main() -> None:
    for path in export_schemas():
        print(path)


if __name__ == "__main__":
    main()
