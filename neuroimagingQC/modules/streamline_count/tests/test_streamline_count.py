"""
Tests for the streamline_count module.
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
    if "streamline_count" not in config.sp:
        config.update_dict(config.sp, {"streamline_count":
                                       {"fn": "*__sc.txt"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Create streamline count files with various values
    values = {
        "sub-S1__sc.txt": "8337903",
        "sub-S2__sc.txt": "9123456",
        "sub-S3__sc.txt": "7654321",
        "sub-S4__sc.txt": "8888888",
        "sub-S5__sc.txt": "9999999",
    }

    for filename, value in values.items():
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w") as f:
            f.write(value)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the streamline_count module can be imported."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    assert hasattr(streamline_count, "MultiqcModule")


def test_parse_single_file(reset_multiqc):
    """Test parsing a single streamline count file."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    # Create a mock file object
    file_content = "8337903"

    f = {
        "f": file_content,
        "fn": "sub-TEST001__sc.txt",
        "s_name": "sub-TEST001",
    }

    # Create a minimal module instance to access parse method
    module = object.__new__(streamline_count.MultiqcModule)
    module.clean_s_name = lambda x, y: x  # Mock clean_s_name method
    result = module.parse_sc_file(f)

    # Check that the file was parsed correctly
    assert "sample_name" in result
    assert "sc_value" in result
    assert result["sample_name"] == "sub-TEST001"
    assert result["sc_value"] == 8337903


def test_iqr_calculation(reset_multiqc):
    """Test IQR-based outlier detection with known outlier.

    Creates test data where one sample is a clear outlier and verifies
    the module correctly identifies it as failing.
    """
    from neuroimagingQC.modules.streamline_count import streamline_count

    tmpdir = tempfile.mkdtemp()

    try:
        # Create test data with known outliers
        # Values: [100, 200, 300, 400, 500, 5000]
        # Q1=200, Q3=400, IQR=200
        # Lower bound = 200 - 3*200 = -400
        # Upper bound = 400 + 3*200 = 1000
        # So sample6 (5000) should be an outlier
        test_values = {
            "sub-sample1__sc.txt": "100",
            "sub-sample2__sc.txt": "200",
            "sub-sample3__sc.txt": "300",
            "sub-sample4__sc.txt": "400",
            "sub-sample5__sc.txt": "500",
            "sub-sample6__sc.txt": "5000",  # Outlier
        }

        for filename, value in test_values.items():
            file_path = os.path.join(tmpdir, filename)
            with open(file_path, "w") as f:
                f.write(value)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["streamline_count"] = [
            {
                "fn": os.path.join(tmpdir, fn),
                "root": tmpdir,
                "s_name": fn.replace("__sc.txt", ""),
                "sp_key": "streamline_count",
            }
            for fn in test_values.keys()
        ]

        module = streamline_count.MultiqcModule()

        # Check that sample6 failed (outlier) and others passed
        section = module.sections[0]
        assert '"sub-sample6": "fail"' in section.status_bar_html
        assert '"sub-sample1": "pass"' in section.status_bar_html
        assert '"sub-sample2": "pass"' in section.status_bar_html
        assert '"sub-sample3": "pass"' in section.status_bar_html
        assert '"sub-sample4": "pass"' in section.status_bar_html
        assert '"sub-sample5": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_ignore_samples_validation(reset_multiqc, test_data_dir):
    """Test that ignored samples are excluded from output."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-S1"]
    config.preserve_module_raw_data = True

    report.files["streamline_count"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-S1__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S1",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S2__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S2",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S3__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S3",
            "sp_key": "streamline_count",
        },
    ]

    module = streamline_count.MultiqcModule()
    assert module is not None

    # Verify that the ignored sample was actually filtered out
    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]
    assert "sub-S1" not in general_stats
    assert "sub-S2" in general_stats
    assert "sub-S3" in general_stats

    # Verify only 2 samples remain after filtering
    assert len(general_stats) == 2

    config.sample_names_ignore = []


def test_data_written_to_general_stats(reset_multiqc, test_data_dir):
    """Test that streamline count data is added to general statistics."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["streamline_count"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-S1__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S1",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S2__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S2",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S3__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S3",
            "sp_key": "streamline_count",
        },
    ]

    module = streamline_count.MultiqcModule()
    assert module is not None

    # Check that general stats were added
    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]

    # Check all three samples are present
    assert len(general_stats) == 3
    assert "sub-S1" in general_stats
    assert "sub-S2" in general_stats
    assert "sub-S3" in general_stats

    # Check that streamline_count field exists with correct values
    s1_row = general_stats["sub-S1"][0]
    s2_row = general_stats["sub-S2"][0]
    s3_row = general_stats["sub-S3"][0]
    assert "streamline_count" in s1_row.data
    assert s1_row.data["streamline_count"] == 8337903
    assert s2_row.data["streamline_count"] == 9123456
    assert s3_row.data["streamline_count"] == 7654321


def test_section_added(reset_multiqc, test_data_dir):
    """Test that a section with plot is added to the report."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["streamline_count"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-S1__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S1",
            "sp_key": "streamline_count",
        }
    ]

    module = streamline_count.MultiqcModule()

    # Check that section was added
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check section properties
    section = module.sections[0]
    assert section.name == "Streamline Count Quality"
    assert section.anchor == "streamline_count_quality"
    assert hasattr(section, "status_bar_html")
    assert hasattr(section, "plot")


def test_configurable_iqr_multiplier(reset_multiqc, test_data_dir):
    """Test that custom IQR multiplier can be configured."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    # Set custom IQR multiplier to 1 (tighter bounds)
    config.streamline_count = {"iqr_multiplier": 1}
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["streamline_count"] = [
        {
            "fn": os.path.join(test_data_dir, "sub-S1__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S1",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S2__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S2",
            "sp_key": "streamline_count",
        },
        {
            "fn": os.path.join(test_data_dir, "sub-S3__sc.txt"),
            "root": test_data_dir,
            "s_name": "sub-S3",
            "sp_key": "streamline_count",
        },
    ]

    module = streamline_count.MultiqcModule()

    # Check that the description includes the custom multiplier
    # Note: The description uses HTML <em> tags which render as "*"
    section = module.sections[0]
    assert "Q1 - 1<em>IQR" in section.description

    # Cleanup config
    delattr(config, "streamline_count")


def test_single_sample_handling(reset_multiqc):
    """Test that the module handles single-sample files correctly.

    When there's only one sample, IQR calculation cannot determine
    outliers, so the module should handle this gracefully.
    """
    from neuroimagingQC.modules.streamline_count import streamline_count

    tmpdir = tempfile.mkdtemp()

    try:
        # Create single-sample file
        single_path = os.path.join(tmpdir, "sub-SINGLE__sc.txt")
        with open(single_path, "w") as f:
            f.write("8500000")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["streamline_count"] = [
            {
                "fn": single_path,
                "root": tmpdir,
                "s_name": "sub-SINGLE",
                "sp_key": "streamline_count",
            }
        ]

        # Module should not crash with single sample
        module = streamline_count.MultiqcModule()
        assert module is not None

        # Check that sections were added
        assert len(module.sections) > 0

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]
        assert "sub-SINGLE" in general_stats

        # Single sample should have pass status (no outliers by definition)
        section = module.sections[0]
        assert '"sub-SINGLE": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "sub-EMPTY__sc.txt")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["streamline_count"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "sub-EMPTY",
                "sp_key": "streamline_count",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            streamline_count.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)


def test_malformed_file_handling(reset_multiqc):
    """Test handling of malformed streamline count files."""
    from neuroimagingQC.modules.streamline_count import streamline_count

    tmpdir = tempfile.mkdtemp()

    try:
        bad_path = os.path.join(tmpdir, "sub-BAD__sc.txt")
        with open(bad_path, "w") as f:
            f.write("not a number\n")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["streamline_count"] = [
            {
                "fn": bad_path,
                "root": tmpdir,
                "s_name": "sub-BAD",
                "sp_key": "streamline_count",
            }
        ]

        # Module should raise exception for malformed file
        with pytest.raises(ModuleNoSamplesFound):
            streamline_count.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
