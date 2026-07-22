"""Typed registry + provenance models for the helmet training pipeline (H1).

Dev-time training infrastructure. It lives in ``experiments/`` and is **not** part
of the ``trafficpulse`` runtime package -- it never ships in the wheel, adds no ML
dependency, and downloads nothing. It uses pydantic only (already a base project
dependency), mirroring the frozen+strict contract style of
``trafficpulse.contracts`` without importing it (the training layer stays
independently movable).

What is modelled
----------------
* **Licence provenance** (:class:`LicenseInfo`) with an explicit verification state
  -- the single most important model here, because the whole helmet effort has
  repeatedly stalled on unverified licence claims. A licence cannot be marked
  ``verified`` without recording the source that verified it.
* **Archive/source provenance** (:class:`ArchiveMetadata`, :class:`DatasetSource`)
  with checksum + download-timestamp **placeholders** (``None`` until a later unit
  actually downloads -- H1 downloads nothing).
* **Dataset entries + registry** (:class:`DatasetEntry`, :class:`DatasetRegistry`)
  with unique-id, schema-version, and licence validation.
* **Corpus version** (:class:`CorpusVersion`) -- a deterministic identity for a
  merged training set, so an exported model can name exactly the data it saw.

Determinism
-----------
Every model is frozen + strict. :meth:`DatasetRegistry.canonical_json` /
:meth:`CorpusVersion.canonical_json` sort their members by id before serialising,
so the JSON (and its ``content_hash``) is a pure function of content, independent
of declaration order.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from .errors import (
    DatasetNotFoundError,
    DuplicateDatasetIdError,
    InvalidLicenseError,
    UnsupportedRegistryVersionError,
)

# --- constrained scalar aliases ----------------------------------------------
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

# --- registry schema versioning ----------------------------------------------
CURRENT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0.0"})


class _Model(BaseModel):
    """Frozen + strict base for every training-registry model."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- licence -----------------------------------------------------------------
class LicenseId(StrEnum):
    """Closed set of licences the pipeline recognises."""

    CC_BY_4_0 = "CC-BY-4.0"
    CC0_1_0 = "CC0-1.0"
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    BSD_3_CLAUSE = "BSD-3-Clause"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"  # non-commercial: NOT permissive
    PROPRIETARY = "proprietary"  # gated / agreement-bound (e.g. AI City)
    UNKNOWN = "unknown"


# Licences permitting commercial use + modification + redistribution.
PERMISSIVE_LICENSES: frozenset[LicenseId] = frozenset(
    {
        LicenseId.CC_BY_4_0,
        LicenseId.CC0_1_0,
        LicenseId.MIT,
        LicenseId.APACHE_2_0,
        LicenseId.BSD_3_CLAUSE,
    }
)
# Licences whose only obligation, once permissive, is crediting the creator.
ATTRIBUTION_REQUIRED: frozenset[LicenseId] = frozenset({LicenseId.CC_BY_4_0})


class VerificationStatus(StrEnum):
    """Whether the project has checked the licence against an authoritative source."""

    UNVERIFIED = "unverified"  # recalled/assumed; never trust for use
    VERIFIED = "verified"  # confirmed from a recorded source
    REJECTED = "rejected"  # checked and found not usable (documents exclusion)


class LicenseInfo(_Model):
    """Licence provenance with an explicit, substantiated verification state."""

    license_id: LicenseId
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_on: date | None = None
    verification_source: str | None = None
    attribution: str | None = None

    @model_validator(mode="after")
    def _substantiated(self) -> Self:
        if self.verification_status is VerificationStatus.VERIFIED:
            if not (self.verification_source and self.verification_source.strip()):
                raise InvalidLicenseError(
                    f"licence {self.license_id.value!r} is marked verified but records no "
                    "verification_source; a verified licence must record how it was checked"
                )
            if self.license_id in ATTRIBUTION_REQUIRED and not (
                self.attribution and self.attribution.strip()
            ):
                raise InvalidLicenseError(
                    f"licence {self.license_id.value!r} requires attribution but none is set"
                )
        return self

    @property
    def is_permissive(self) -> bool:
        """Whether the licence permits commercial use, modification, and redistribution."""

        return self.license_id in PERMISSIVE_LICENSES

    @property
    def is_usable(self) -> bool:
        """Whether the pipeline may train on this dataset: permissive AND verified."""

        return self.verification_status is VerificationStatus.VERIFIED and self.is_permissive


# --- archive / source provenance ---------------------------------------------
class ArchiveMetadata(_Model):
    """Provenance of a dataset's downloadable archive (placeholders until fetched).

    ``sha256`` and ``downloaded_at`` are ``None`` until a later unit performs the
    (out-of-scope for H1) download; ``None`` honestly means "not yet acquired",
    never a fabricated value.
    """

    filename: NonEmptyStr
    sha256: Sha256Hex | None = None
    size_bytes: int | None = None
    downloaded_at: datetime | None = None

    @model_validator(mode="after")
    def _non_negative_size(self) -> Self:
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        return self


class DatasetSource(_Model):
    """Where a dataset comes from and how it is retrieved."""

    url: NonEmptyStr
    retrieval_method: NonEmptyStr  # "osf" | "roboflow" | "manual" | ...
    accessed_at: datetime | None = None
    notes: str | None = None


# --- dataset entry + registry ------------------------------------------------
class DatasetEntry(_Model):
    """One registered dataset: identity, provenance, licence, and expected files."""

    dataset_id: Slug
    name: NonEmptyStr
    version: NonEmptyStr
    modality: NonEmptyStr  # "video" | "images"
    domain: NonEmptyStr  # e.g. "myanmar-traffic"
    source: DatasetSource
    license: LicenseInfo
    archive: ArchiveMetadata | None = None
    required_paths: tuple[str, ...] = ()  # relative to the dataset's raw dir
    description: str | None = None


class DatasetRegistry(_Model):
    """A validated collection of dataset entries keyed by ``dataset_id``."""

    schema_version: str = CURRENT_SCHEMA_VERSION
    entries: tuple[DatasetEntry, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedRegistryVersionError(
                f"registry schema_version {self.schema_version!r} is not supported "
                f"(supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
            )
        seen: set[str] = set()
        for entry in self.entries:
            if entry.dataset_id in seen:
                raise DuplicateDatasetIdError(f"duplicate dataset_id: {entry.dataset_id!r}")
            seen.add(entry.dataset_id)
        return self

    def get(self, dataset_id: str) -> DatasetEntry:
        """Return the entry for ``dataset_id`` or raise :class:`DatasetNotFoundError`."""

        for entry in self.entries:
            if entry.dataset_id == dataset_id:
                return entry
        raise DatasetNotFoundError(
            f"no dataset {dataset_id!r} in registry (known: {list(self.ids())})"
        )

    def ids(self) -> tuple[str, ...]:
        """All dataset ids, sorted."""

        return tuple(sorted(entry.dataset_id for entry in self.entries))

    def usable(self) -> tuple[DatasetEntry, ...]:
        """Entries whose licence is verified-permissive, sorted by id."""

        return tuple(
            sorted(
                (e for e in self.entries if e.license.is_usable),
                key=lambda e: e.dataset_id,
            )
        )

    def canonical_json(self) -> str:
        """Deterministic JSON: entries sorted by id (order-independent)."""

        ordered = self.model_copy(
            update={"entries": tuple(sorted(self.entries, key=lambda e: e.dataset_id))}
        )
        return ordered.model_dump_json()

    def content_hash(self) -> str:
        """SHA-256 over :meth:`canonical_json` -- a stable identity for the registry."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --- corpus version ----------------------------------------------------------
class CorpusMember(_Model):
    """One dataset pinned at a version, as a member of a training corpus."""

    dataset_id: Slug
    dataset_version: NonEmptyStr


class CorpusVersion(_Model):
    """A deterministic identity for a merged training corpus.

    Lets an exported model name exactly the datasets (and their versions) it was
    trained on. :meth:`content_hash` is order-independent, so the same member set
    always yields the same identity.
    """

    corpus_id: Slug
    version: NonEmptyStr
    members: tuple[CorpusMember, ...]
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.members:
            raise ValueError("a corpus version must have at least one member")
        seen: set[str] = set()
        for member in self.members:
            if member.dataset_id in seen:
                raise DuplicateDatasetIdError(
                    f"duplicate corpus member dataset_id: {member.dataset_id!r}"
                )
            seen.add(member.dataset_id)
        return self

    @classmethod
    def from_registry(
        cls, registry: DatasetRegistry, *, corpus_id: str, version: str
    ) -> CorpusVersion:
        """Build a corpus from every **usable** dataset in ``registry``.

        Only verified-permissive datasets are included -- a corpus can never
        silently pin a dataset the pipeline is not cleared to train on.
        """

        members = tuple(
            CorpusMember(dataset_id=e.dataset_id, dataset_version=e.version)
            for e in registry.usable()
        )
        return cls(corpus_id=corpus_id, version=version, members=members)

    def canonical_json(self) -> str:
        """Deterministic JSON: members sorted by ``(dataset_id, dataset_version)``."""

        ordered = self.model_copy(
            update={
                "members": tuple(
                    sorted(self.members, key=lambda m: (m.dataset_id, m.dataset_version))
                )
            }
        )
        return ordered.model_dump_json()

    def content_hash(self) -> str:
        """SHA-256 over :meth:`canonical_json` -- a stable identity for the corpus."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
