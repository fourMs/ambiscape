"""The package version has one source, and the build reads it from there.

``__version__`` drifted from the packaged version through three releases
(0.23.x, 0.24.0, 0.24.1): each bump edited ``pyproject.toml`` and left the
module attribute behind, so an install of 0.24.1 reported 0.24.0 to anyone
who asked it. Reports in this project cite the toolbox by version, which
makes a package that misstates its own version a correctness problem
rather than a cosmetic one.

The fix is that ``src/ambiscape/__init__.py`` holds the number and
setuptools reads it from there. These tests keep it that way.

The checks read ``pyproject.toml`` with a small section scanner rather
than a TOML parser, because ``tomllib`` is standard only from Python 3.11
and this package supports 3.10. What is needed here is whether a key is
present in a section, which does not warrant a dependency.
"""
import re
from pathlib import Path

import ambiscape

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _section(name: str) -> list[str]:
    """Non-comment, non-blank lines of one top-level ``[section]``."""
    out, inside = [], False
    for raw in PYPROJECT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            inside = line == f"[{name}]"
            continue
        if inside and line and not line.startswith("#"):
            out.append(line)
    return out


def _value(section: str, key: str) -> str | None:
    for line in _section(section):
        m = re.match(rf"{re.escape(key)}\s*=\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return None


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?\w+)?", ambiscape.__version__), \
        f"__version__ is not a version string: {ambiscape.__version__!r}"


def test_pyproject_declares_no_second_version():
    """A static version in pyproject.toml is how the drift happened."""
    assert _value("project", "version") is None, (
        "pyproject.toml carries its own version again. It must stay dynamic, "
        "or the two numbers will drift as they did through 0.23.x-0.24.1."
    )
    dynamic = _value("project", "dynamic") or ""
    assert "version" in dynamic, \
        "pyproject.toml should declare version in [project].dynamic"


def test_build_reads_the_module_attribute():
    attr = _value("tool.setuptools.dynamic", "version") or ""
    assert "ambiscape.__version__" in attr, (
        f"the build resolves the version from {attr!r}; it should read "
        "ambiscape.__version__ so there is exactly one place to edit"
    )


def test_setuptools_resolves_the_declared_version():
    """The number setuptools would package equals the one the module reports."""
    from setuptools.config.pyprojecttoml import read_configuration

    resolved = read_configuration(str(PYPROJECT))["project"].get("version")
    assert resolved == ambiscape.__version__, (
        f"build would package {resolved!r} while the module reports "
        f"{ambiscape.__version__!r}"
    )
