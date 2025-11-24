"""
Tests for the framewise_displacement module.
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
    if "framewise_displacement" not in config.sp:
        config.update_dict(
            config.sp,
            {"framewise_displacement": {"fn": "*eddy_restricted_movement_rms.txt"}},
        )
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Create FD files with different threshold values
    fd_pass = """0.840188 0
0.394383 0.12
0.683099 0.08
0.59844 0.09
0.711647 0.04
"""

    fd_warn = """0.840188 0
0.894383 1.28
1.183099 0.82
1.29844 1.09
0.911647 1.04
"""

    fd_fail = """0.840188 0
2.394383 2.12
3.183099 2.82
2.59844 2.29
2.911647 2.54
"""

    # Create files
    pass_path = os.path.join(tmpdir, "sub-PASS001_eddy_restricted_movement_rms.txt")
    with open(pass_path, "w") as f:
        f.write(fd_pass)

    warn_path = os.path.join(tmpdir, "sub-WARN001_eddy_restricted_movement_rms.txt")
    with open(warn_path, "w") as f:
        f.write(fd_warn)

    fail_path = os.path.join(tmpdir, "sub-FAIL001_eddy_restricted_movement_rms.txt")
    with open(fail_path, "w") as f:
        f.write(fd_fail)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the framewise_displacement module can be imported."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    assert hasattr(framewise_displacement, "MultiqcModule")


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-PASS001"]

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "framewise_displacement",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "framewise_displacement",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "framewise_displacement",
        },
    ]

    module = framewise_displacement.MultiqcModule()
    assert module is not None

    # Verify that the ignored sample was actually filtered out
    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]
    assert "sub-PASS001" not in general_stats
    assert "sub-WARN001" in general_stats
    assert "sub-FAIL001" in general_stats

    # Verify only 2 samples remain after filtering
    assert len(general_stats) == 2

    config.sample_names_ignore = []


def test_parse_fd_file(reset_multiqc):
    """Test parsing a single FD file."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    # Create a mock file object
    file_content = """0.840188 0.0
0.394383 0.12
0.683099 0.08
0.59844 0.09
0.711647 0.04
"""

    f = {
        "f": file_content,
        "fn": "sub-TEST001_eddy_restricted_movement_rms.txt",
        "s_name": "sub-TEST001",
    }

    # Create a minimal module instance to access parse method
    module = object.__new__(framewise_displacement.MultiqcModule)
    module.clean_s_name = lambda x, y: x  # Mock clean_s_name method
    result = module.parse_fd_file(f)

    # Check that the file was parsed correctly
    assert "sample_name" in result
    assert "values" in result
    assert result["sample_name"] == "sub-TEST001"
    assert len(result["values"]) == 5
    assert abs(result["values"][0] - 0.0) < 0.0001
    assert abs(result["values"][1] - 0.12) < 0.0001


def test_status_assignment_pass(reset_multiqc, test_data_dir):
    """Test that PASS status is assigned correctly (max FD < 0.8)."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "framewise_displacement",
        }
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that the sample has pass status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    assert hasattr(section, "status_bar_html")
    assert '"sub-PASS001": "pass"' in section.status_bar_html
    assert "bg-success" in section.status_bar_html


def test_status_assignment_warn(reset_multiqc, test_data_dir):
    """Test that WARN status is assigned correctly (0.8 <= max FD < 2.0)."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "framewise_displacement",
        }
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that the sample has warn status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    assert hasattr(section, "status_bar_html")
    assert '"sub-WARN001": "warn"' in section.status_bar_html
    assert "bg-warning" in section.status_bar_html


def test_status_assignment_fail(reset_multiqc, test_data_dir):
    """Test that FAIL status is assigned correctly (max FD >= 2.0)."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "framewise_displacement",
        }
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that the sample has fail status in the status bar HTML
    assert len(module.sections) > 0
    section = module.sections[0]
    assert hasattr(section, "status_bar_html")
    assert '"sub-FAIL001": "fail"' in section.status_bar_html
    assert "bg-danger" in section.status_bar_html


def test_data_written_to_general_stats(reset_multiqc, test_data_dir):
    """Test that max FD data is added to general statistics."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "framewise_displacement",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "framewise_displacement",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "framewise_displacement",
        },
    ]

    module = framewise_displacement.MultiqcModule()
    assert module is not None

    # Check that general stats were added
    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]

    # Check all three samples are present
    assert len(general_stats) == 3
    assert "sub-PASS001" in general_stats
    assert "sub-WARN001" in general_stats
    assert "sub-FAIL001" in general_stats

    # Check that max_fd field exists with reasonable values
    pass_row = general_stats["sub-PASS001"][0]
    warn_row = general_stats["sub-WARN001"][0]
    fail_row = general_stats["sub-FAIL001"][0]
    assert "max_fd" in pass_row.data
    # PASS should have max FD < 0.8
    assert pass_row.data["max_fd"] < 0.8
    # WARN should have max FD between 0.8 and 2.0
    assert 0.8 <= warn_row.data["max_fd"] < 2.0
    # FAIL should have max FD >= 2.0
    assert fail_row.data["max_fd"] >= 2.0


def test_multi_subject_section_added(reset_multiqc, test_data_dir):
    """Test that section with plot is added in multi-subject mode."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "framewise_displacement",
        }
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that section was added
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check section properties
    section = module.sections[0]
    assert section.name == "Framewise Displacement"
    assert section.anchor == "fd_multi_subject"
    assert hasattr(section, "status_bar_html")
    assert hasattr(section, "plot")


def test_single_subject_mode(reset_multiqc, test_data_dir):
    """Test that single-subject mode creates appropriate section."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": True}

    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-PASS001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-PASS001",
            "sp_key": "framewise_displacement",
        }
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that section was added
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check section properties for single-subject mode
    section = module.sections[0]
    assert section.name == "Framewise Displacement"
    assert section.anchor == "fd_single_subject"
    assert hasattr(section, "plot")

    # Single-subject mode should NOT have status bar or general stats
    assert not hasattr(section, "status_bar_html") or section.status_bar_html == ""
    assert len(report.general_stats_data) == 0


def test_configurable_thresholds(reset_multiqc, test_data_dir):
    """Test that custom thresholds can be configured."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    # Set custom thresholds: warn=1.5, fail=2.5
    config.framewise_displacement = {"warn_threshold": 1.5, "fail_threshold": 2.5}
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    # sub-WARN001 has max FD ~1.29, should be pass with threshold 1.5
    # sub-FAIL001 has max FD ~3.18, should be fail with threshold 2.5
    report.files["framewise_displacement"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-WARN001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-WARN001",
            "sp_key": "framewise_displacement",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-FAIL001_eddy_restricted_movement_rms.txt"),
            "root": test_data_dir,
            "s_name": "sub-FAIL001",
            "sp_key": "framewise_displacement",
        },
    ]

    module = framewise_displacement.MultiqcModule()

    # Check that statuses reflect custom thresholds
    section = module.sections[0]
    # sub-WARN001 should be pass with threshold 1.5
    assert '"sub-WARN001": "pass"' in section.status_bar_html
    # sub-FAIL001 should be fail with threshold 2.5
    assert '"sub-FAIL001": "fail"' in section.status_bar_html

    # Cleanup config
    delattr(config, "framewise_displacement")


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "sub-EMPTY_eddy_restricted_movement_rms.txt")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["framewise_displacement"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "sub-EMPTY",
                "sp_key": "framewise_displacement",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            framewise_displacement.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)


def test_malformed_file_handling(reset_multiqc):
    """Test handling of malformed FD files."""
    from neuroimaging.modules.framewise_displacement import framewise_displacement

    tmpdir = tempfile.mkdtemp()

    try:
        bad_path = os.path.join(tmpdir, "sub-BAD_eddy_restricted_movement_rms.txt")
        with open(bad_path, "w") as f:
            f.write("not valid data\nmore invalid stuff\n")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["framewise_displacement"] = [
            {
                "fn": bad_path,
                "root": tmpdir,
                "s_name": "sub-BAD",
                "sp_key": "framewise_displacement",
            }
        ]

        # Module should raise exception for malformed file
        with pytest.raises(ModuleNoSamplesFound):
            framewise_displacement.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
