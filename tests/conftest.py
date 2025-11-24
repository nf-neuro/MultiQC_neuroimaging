"""
Pytest configuration for MultiQC-neuroimaging tests.
"""

import pytest

from multiqc import report, config


@pytest.fixture(autouse=True)
def reset():
    """
    Reset MultiQC session after each test: reset config and report
    """
    yield
    report.reset()
    config.reset()
