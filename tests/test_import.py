"""Import and version smoke tests for the TrafficPulse package root (U1).

These verify only the U1 acceptance criteria:
- ``import trafficpulse`` succeeds;
- a version attribute exists;
- the version value is a non-empty string.
"""

import trafficpulse


def test_import_succeeds() -> None:
    assert trafficpulse is not None


def test_version_attribute_exists() -> None:
    assert hasattr(trafficpulse, "__version__")


def test_version_is_non_empty_string() -> None:
    version = trafficpulse.__version__
    assert isinstance(version, str)
    assert version.strip() != ""
