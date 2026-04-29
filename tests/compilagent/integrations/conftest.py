"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest

from compilagent.core.backend import backend_registry
from compilagent.core.workload_registry import workload_registry
from compilagent.harness.registry import harness_registry


@pytest.fixture(autouse=True)
def _reset_registries():
    backend_registry.clear()
    workload_registry.clear()
    harness_registry.clear()
    yield
    backend_registry.clear()
    workload_registry.clear()
    harness_registry.clear()
