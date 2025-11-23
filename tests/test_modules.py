"""
Tests for neuroimagingQC plugin modules.

This file contains tests to verify that all neuroimagingQC modules
work correctly and handle edge cases properly.
"""

import tempfile

import pytest

from multiqc import BaseMultiqcModule, config, report

# List of all neuroimagingQC modules
modules = [
    ("tractometry", "neuroimagingQC.modules.tractometry:MultiqcModule"),
    ("cortical", "neuroimagingQC.modules.cortical:MultiqcModule"),
    ("subcortical", "neuroimagingQC.modules.subcortical:MultiqcModule"),
    (
        "framewise_displacement",
        "neuroimagingQC.modules.framewise_displacement:MultiqcModule",
    ),
    ("coverage", "neuroimagingQC.modules.coverage:MultiqcModule"),
    (
        "streamline_count",
        "neuroimagingQC.modules.streamline_count:MultiqcModule",
    ),
]


@pytest.fixture(autouse=True)
def reset_report():
    """Reset report state after each test."""
    report.reset()
    yield
    report.reset()


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config state after each test."""
    original_strict = config.strict
    original_sample_names_ignore = config.sample_names_ignore[:]
    original_single_subject = config.kwargs.get("single_subject", False)
    yield
    config.strict = original_strict
    config.sample_names_ignore[:] = original_sample_names_ignore
    if "single_subject" in config.kwargs:
        config.kwargs["single_subject"] = original_single_subject


@pytest.mark.parametrize("module_id,module_path", modules)
def test_module_loads(module_id, module_path):
    """Verify that each module can be imported successfully."""
    try:
        module_parts = module_path.split(":")
        module_name = module_parts[0]
        class_name = module_parts[1]

        exec(f"from {module_name} import {class_name}")
    except ImportError as e:
        pytest.fail(f"Failed to import {module_id}: {e}")


def test_write_data_file(monkeypatch, tmp_path):
    """
    Test module.write_data_file() writes something
    """
    (tmp_path / "multiqc_tmp").mkdir()
    monkeypatch.setattr(
        tempfile, "mkdtemp", lambda: tmp_path / "multiqc_tmp"
    )

    module = BaseMultiqcModule()
    module.write_data_file(
        {"Sample": {"key": "value"}}, "multiqc_test_module"
    )

    expected_path = (
        tmp_path / "multiqc_tmp" / "multiqc_data" / "multiqc_test_module.txt"
    )
    assert expected_path.exists()
    expected_content = """Sample\tkey\nSample\tvalue""".strip()
    assert expected_path.open().read().strip() == expected_content


def test_sample_name_cleaning():
    """Test that sample name cleaning works correctly."""
    module = BaseMultiqcModule()

    # Test cleaning of common neuroimaging patterns
    assert "sub-1019" in module._clean_s_name("sub-1019__dice.txt")
    fd_file = "sub-1019_eddy_restricted_movement_rms.txt"
    assert "sub-1019" in module._clean_s_name(fd_file)
    assert "sub-1019" in module._clean_s_name("sub-1019__sc.txt")
