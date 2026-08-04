"""Production logging configuration + request correlation (H16)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest
from _app_helpers import make_client, make_config

from trafficpulse.app.config import AppConfig
from trafficpulse.app.logging_config import (
    REQUEST_ID_HEADER,
    ROOT_LOGGER,
    configure_logging,
    current_request_id,
    normalize_log_level,
)


@pytest.fixture(autouse=True)
def _restore_logging() -> object:
    """Leave the package logger as the suite found it."""

    package = logging.getLogger(ROOT_LOGGER)
    handlers = list(package.handlers)
    level, propagate = package.level, package.propagate
    yield
    package.handlers = handlers
    package.setLevel(level)
    package.propagate = propagate


# --- level normalisation ---------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("debug", "DEBUG"),
        ("Info", "INFO"),
        ("WARNING", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
        ("  info  ", "INFO"),
    ],
)
def test_level_names_are_case_and_space_insensitive(given: str, expected: str) -> None:
    assert normalize_log_level(given) == expected


@pytest.mark.parametrize("given", ["verbose", "", "TRACE", "42"])
def test_an_unknown_level_falls_back_rather_than_raising(given: str) -> None:
    """A typo in a deployment variable must not stop the service from starting."""

    assert normalize_log_level(given) == "INFO"


def test_no_level_supplied_uses_the_default() -> None:
    assert normalize_log_level(None) == "INFO"


# --- configuration ---------------------------------------------------------------
def test_configuring_emits_formatted_records_with_subsystem_and_time() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("trafficpulse.recovery").info("recovered %d run(s)", 3)

    line = stream.getvalue()
    assert "recovered 3 run(s)" in line
    assert "INFO" in line
    assert "trafficpulse.recovery" in line  # subsystem, not module path
    assert line.startswith("20")  # ISO timestamp


def test_info_is_emitted_not_dropped() -> None:
    """The H16 defect: with no configuration, INFO went nowhere.

    The startup recovery report -- the one line describing what a repository
    actually contained -- was invisible in production because of it.
    """

    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("trafficpulse.app").info("visible")
    assert "visible" in stream.getvalue()


def test_the_level_filters_as_configured() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    logger = logging.getLogger("trafficpulse.app")
    logger.info("quiet")
    logger.warning("loud")

    output = stream.getvalue()
    assert "quiet" not in output
    assert "loud" in output


def test_debug_is_reachable() -> None:
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    logging.getLogger("trafficpulse.evidence").debug("detail")
    assert "detail" in stream.getvalue()


def test_configuring_twice_does_not_duplicate_output() -> None:
    """Building several applications in one process must not multiply log lines."""

    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    configure_logging("INFO", stream=stream)
    logging.getLogger("trafficpulse.app").info("once")
    assert stream.getvalue().count("once") == 1


def test_configuration_is_scoped_to_the_package() -> None:
    """A library must not hijack the host application's root logger."""

    root_handlers = list(logging.getLogger().handlers)
    configure_logging("DEBUG", stream=io.StringIO())
    assert logging.getLogger().handlers == root_handlers
    assert logging.getLogger().level == logging.getLogger().level


def test_returns_the_level_actually_applied() -> None:
    assert configure_logging("nonsense", stream=io.StringIO()) == "INFO"
    assert configure_logging("error", stream=io.StringIO()) == "ERROR"


# --- config plumbing -------------------------------------------------------------
def test_log_level_is_read_from_the_environment() -> None:
    config = AppConfig.from_env({"TRAFFICPULSE_APP_LOG_LEVEL": "debug"})
    assert config.log_level == "DEBUG"


def test_an_invalid_configured_level_degrades_to_info() -> None:
    assert AppConfig.from_env({"TRAFFICPULSE_APP_LOG_LEVEL": "chatty"}).log_level == "INFO"
    assert AppConfig(storage_dir=Path("x"), log_level="loud").log_level == "INFO"


def test_the_default_level_is_info() -> None:
    assert AppConfig.from_env({}).log_level == "INFO"


# --- request correlation ---------------------------------------------------------
def test_outside_a_request_there_is_no_id() -> None:
    """A background job thread legitimately runs outside any request."""

    assert current_request_id() == "-"


def test_every_response_carries_a_request_id(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/health")
    assert response.headers[REQUEST_ID_HEADER]


def test_each_request_gets_a_distinct_id(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    first = client.get("/api/health").headers[REQUEST_ID_HEADER]
    second = client.get("/api/health").headers[REQUEST_ID_HEADER]
    assert first != second


def test_an_inbound_request_id_is_honoured(tmp_path: Path) -> None:
    """An id assigned by a reverse proxy must survive into these logs."""

    client = make_client(tmp_path)
    response = client.get("/api/health", headers={REQUEST_ID_HEADER: "trace-abc"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-abc"


def test_an_error_response_is_still_correlated(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/events/evt-nope")
    assert response.status_code == 404
    assert response.headers[REQUEST_ID_HEADER]


def test_the_request_id_never_enters_the_response_body(tmp_path: Path) -> None:
    """Correlation is a header concern; no API response model changed for it."""

    client = make_client(tmp_path)
    body = client.get("/api/health").json()
    assert "request_id" not in body


def test_the_engine_log_path_sits_outside_the_run_tree(tmp_path: Path) -> None:
    """A process-level stream must not live inside the write-once run tree."""

    config = make_config(tmp_path)
    assert config.engine_log_path.name == "engine.jsonl"
    assert config.runs_dir not in config.engine_log_path.parents
