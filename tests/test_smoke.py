"""Smoke tests: verify the package installs and imports cleanly."""

import lead_priority


def test_package_exposes_version() -> None:
    assert isinstance(lead_priority.__version__, str)
    assert lead_priority.__version__
