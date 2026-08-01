# ABOUTME: Smoke test proving the teardown_lab package imports and exposes a version.
# ABOUTME: Runs headless; no game, no X display.

import teardown_lab


def test_package_imports():
    assert teardown_lab.__version__ == "0.1.0"
