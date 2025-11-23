"""
Tests for the coverage module.
"""

import os
import tempfile
import shutil
import pytest
from multiqc import config, report
from multiqc.base_module import ModuleNoSamplesFound


@pytest.fixture
def reset_multiqc():
    """Reset MultiQC state before each test."""
    config.reset()
    report.reset()
    # Register search patterns after reset
    if "coverage" not in config.sp:
        config.update_dict(config.sp, {"coverage": {"fn": "*dice.txt"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Create dice coefficient files with different threshold values
    values = {
        "sub-PASS001__dice.txt": "0.9532",
        "sub-WARN001__dice.txt": "0.8593",
        "sub-FAIL001__dice.txt": "0.7234",
    }

    for filename, value in values.items():
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w") as f:
            f.write(value)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the coverage module can be imported."""
    from neuroimagingQC.modules.coverage import coverage

    assert hasattr(coverage, "MultiqcModule")


def test_parse_dice_file(reset_multiqc):
    """Test parsing a single dice coefficient file."""
    from neuroimagingQC.modules.coverage import coverage

    # Create a mock file object with dice coefficient
    file_content = "0.8593300982298845"

    f = {
        "f": file_content,
        "fn": "sub-TEST001__dice.txt",
        "s_name": "sub-TEST001",
    }

    # Create a minimal module instance to access parse method
    module = object.__new__(coverage.MultiqcModule)
    module.clean_s_name = lambda x, y: x  # Mock clean_s_name method
    result = module.parse_dice_file(f)

    # Check that the file was parsed correctly
    assert "sample_name" in result
    assert "dice_value" in result
    assert result["sample_name"] == "sub-TEST001"
    assert abs(result["dice_value"] - 0.8593) < 0.0001


def test_status_assignment_pass(reset_multiqc, test_data_dir):
    """Test that PASS status is assigned correctly (dice >= 0.9)."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "coverage",
        }
    ]

    module = coverage.MultiqcModule()

    # Check that the sample has pass status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    # Status info is embedded in status_bar_html
    assert hasattr(section, "status_bar_html")
    assert '"sub-PASS001": "pass"' in section.status_bar_html
    assert "bg-success" in section.status_bar_html


def test_status_assignment_warn(reset_multiqc, test_data_dir):
    """Test that WARN status is assigned correctly (0.8 <= dice < 0.9)."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "coverage",
        }
    ]

    module = coverage.MultiqcModule()

    # Check that the sample has warn status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    # Status info is embedded in status_bar_html
    assert hasattr(section, "status_bar_html")
    assert '"sub-WARN001": "warn"' in section.status_bar_html
    assert "bg-warning" in section.status_bar_html


def test_status_assignment_fail(reset_multiqc, test_data_dir):
    """Test that FAIL status is assigned correctly (dice < 0.8)."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "coverage",
        }
    ]

    module = coverage.MultiqcModule()

    # Check that the sample has fail status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    # Status info is embedded in status_bar_html
    assert hasattr(section, "status_bar_html")
    assert '"sub-FAIL001": "fail"' in section.status_bar_html
    assert "bg-danger" in section.status_bar_html


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test that ignored samples are excluded from output."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-PASS001"]

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "coverage",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "coverage",
        },
    ]

    module = coverage.MultiqcModule()
    assert module is not None

    # Check that ignored sample is not in general stats
    assert len(report.general_stats_data) > 0
    # general_stats_data is a dict with module IDs as keys
    module_data = list(report.general_stats_data.values())[0]
    stats_samples = list(module_data.keys())
    assert "sub-PASS001" not in stats_samples
    assert "sub-WARN001" in stats_samples

    config.sample_names_ignore = []


def test_data_written_to_general_stats(reset_multiqc, test_data_dir):
    """Test that dice data is added to general statistics."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "coverage",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "coverage",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "coverage",
        },
    ]

    module = coverage.MultiqcModule()
    assert module is not None

    # Check that general stats were added
    assert len(report.general_stats_data) > 0
    # general_stats_data is a dict with module IDs as keys
    general_stats = list(report.general_stats_data.values())[0]

    # Check all three samples are present
    assert len(general_stats) == 3
    assert "sub-PASS001" in general_stats
    assert "sub-WARN001" in general_stats
    assert "sub-FAIL001" in general_stats

    # Check that dice_coefficient field exists and has correct values
    # general_stats values are InputRow objects with data attribute
    pass_row = general_stats["sub-PASS001"][0]
    warn_row = general_stats["sub-WARN001"][0]
    fail_row = general_stats["sub-FAIL001"][0]
    assert "dice_coefficient" in pass_row.data
    assert abs(pass_row.data["dice_coefficient"] - 0.9532) < 0.0001
    assert abs(warn_row.data["dice_coefficient"] - 0.8593) < 0.0001
    assert abs(fail_row.data["dice_coefficient"] - 0.7234) < 0.0001


def test_section_added(reset_multiqc, test_data_dir):
    """Test that a section with plot is added to the report."""
    from neuroimagingQC.modules.coverage import coverage

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "coverage",
        }
    ]

    module = coverage.MultiqcModule()

    # Check that section was added
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check section properties
    section = module.sections[0]
    assert section.name == "Coverage Quality"
    assert section.anchor == "coverage_quality"
    assert hasattr(section, "status_bar_html")
    assert hasattr(section, "plot")


def test_configurable_thresholds(reset_multiqc, test_data_dir):
    """Test that custom thresholds can be configured."""
    from neuroimagingQC.modules.coverage import coverage

    # Set custom thresholds: warn=0.85, fail=0.75
    config.coverage = {"warn_threshold": 0.85, "fail_threshold": 0.75}
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    # sub-FAIL001 has dice=0.7234, should be fail (< 0.75)
    # sub-WARN001 has dice=0.8593, should be pass now (>= 0.85)
    report.files["coverage"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "coverage",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001__dice.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "coverage",
        },
    ]

    module = coverage.MultiqcModule()

    # Check that statuses reflect custom thresholds
    section = module.sections[0]
    # sub-WARN001 (0.8593) should be pass with threshold 0.85
    assert '"sub-WARN001": "pass"' in section.status_bar_html
    # sub-FAIL001 (0.7234) should be fail with threshold 0.75
    assert '"sub-FAIL001": "fail"' in section.status_bar_html

    # Cleanup config
    delattr(config, "coverage")


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimagingQC.modules.coverage import coverage

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "sub-EMPTY__dice.txt")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["coverage"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "sub-EMPTY",
                "sp_key": "coverage",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            coverage.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)


def test_malformed_file_handling(reset_multiqc):
    """Test handling of malformed dice files."""
    from neuroimagingQC.modules.coverage import coverage

    tmpdir = tempfile.mkdtemp()

    try:
        bad_path = os.path.join(tmpdir, "sub-BAD__dice.txt")
        with open(bad_path, "w") as f:
            f.write("not a number\n")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["coverage"] = [
            {
                "fn": bad_path,
                "root": tmpdir,
                "s_name": "sub-BAD",
                "sp_key": "coverage",
            }
        ]

        # Module should raise exception for malformed file
        with pytest.raises(ModuleNoSamplesFound):
            coverage.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
