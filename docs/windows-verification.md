# Windows Verification — U1 Repository Baseline

**Scope:** Native Windows (PowerShell) verification of the Phase 0-F unit U1
repository baseline. CI runs single-environment **Linux**
(`.github/workflows/ci.yml`); this document is how the team confirms the same
quality gates pass on native Windows, and records when they last did.

Run every command from the repository root (the directory containing
`pyproject.toml`) in **Windows PowerShell**.

---

## Prerequisites

- Windows 10/11.
- Python **>= 3.11** on `PATH` (`requires-python = ">=3.11"` in `pyproject.toml`).
- `git` (to obtain the repository).
- No GPU, ML/CV libraries, or datasets are required for U1.

Confirm Python:

```powershell
python --version
```

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, either allow it for the current
process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

or skip activation and call the tools directly via `.\.venv\Scripts\<tool>.exe`
(the exact form used to record the results below).

## 2. Install the project and development tooling

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the package (editable) plus the `dev` extra: ruff, mypy, pytest.

## 3. Ruff (lint)

```powershell
ruff check .
```

## 4. mypy (static type check)

```powershell
mypy src
```

## 5. pytest (tests)

```powershell
pytest -q
```

## 6. Import / version smoke check

```powershell
python -c "import trafficpulse; print(trafficpulse.__version__)"
```

Expected: prints a non-empty version string (currently `0.1.0`).

---

## Recorded execution

Each row is filled in only when the command was actually executed on native
Windows. Do not mark a row complete on assertion alone.

| Field | Value |
|---|---|
| Execution date | 2026-07-07 |
| Executed by | Repository maintainer (native Windows PowerShell session) |
| OS | Windows 11 Home Single Language 10.0.26200 |
| Python | 3.13.3 |
| Tooling versions | ruff 0.15.20, mypy 2.1.0, pytest 9.1.1 |
| Environment | project-local `.venv`, editable install of `trafficpulse` + `[dev]` |

| Check | Command | Result (2026-07-07) |
|---|---|---|
| Install | `pip install -e ".[dev]"` | PASS — `trafficpulse-0.1.0` built and installed |
| Ruff | `ruff check .` | PASS — `All checks passed!` (exit 0) |
| mypy | `mypy src` | PASS — `Success: no issues found in 1 source file` (exit 0) |
| pytest | `pytest -q` | PASS — `3 passed` (exit 0) |
| Import/version | `python -c "import trafficpulse; print(trafficpulse.__version__)"` | PASS — printed `0.1.0` (exit 0) |

**Status:** Native Windows verification of U1 **completed and passing** on the
date above. Re-run and re-date this section whenever the toolchain, Python
version, or U1 files change.
