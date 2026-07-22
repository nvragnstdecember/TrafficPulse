"""The converter interface + shared label-mapping (H2).

An :class:`AnnotationAdapter` turns one dataset's native annotations into unified
objects. The base carries **no** dataset-specific knowledge -- only the abstract
contract and the shared, loud label-mapping helper every adapter uses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path

from ..errors import UnsupportedLabelError
from ..unified import UnifiedClass, UnifiedObject

# A label map value of ``None`` means "recognised, but intentionally skipped"
# (e.g. a licence-plate box); a label ABSENT from the map is an error.
LabelMap = Mapping[str, UnifiedClass | None]


class AnnotationAdapter(ABC):
    """Converts one dataset's native annotations into :class:`UnifiedObject`\\ s."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable adapter id, stamped into every object's provenance."""

    @abstractmethod
    def detect(self, root: Path) -> bool:
        """Whether this adapter recognises the layout under ``root`` (no parsing)."""

    @abstractmethod
    def convert(
        self, root: Path, *, dataset_id: str, dataset_version: str
    ) -> Iterator[UnifiedObject]:
        """Yield unified objects for the dataset rooted at ``root``.

        Deterministic: the same directory yields the same objects in the same
        order. Framework-neutral: no ML dependency, no download.
        """


def map_label(label_map: LabelMap, raw: str, *, adapter: str) -> UnifiedClass | None:
    """Map a native label to a :class:`UnifiedClass`, or ``None`` to skip it.

    Raises :class:`UnsupportedLabelError` for a label absent from ``label_map`` --
    never a silent drop or a guess.
    """

    if raw not in label_map:
        raise UnsupportedLabelError(
            f"adapter {adapter!r} has no mapping for source label {raw!r}; "
            f"known labels: {sorted(label_map)}"
        )
    return label_map[raw]
