"""
Tests for the metricsinroi module.
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
    if "metricsinroi" not in config.sp:
        config.update_dict(config.sp, {"metricsinroi": {"fn": "rois_mean_stats.tsv"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Sample metricsinroi data
    header = "sample\troi\tad\tfa\tmd\trd"
    data = f"""{header}
sub-P1688\tAC\t0.00127\t0.3245\t0.00093\t0.00076
sub-P1688\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058
sub-P1688\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059
sub-P1536\tAC\t0.00130\t0.3350\t0.00095\t0.00078
sub-P1536\tAF_L\t0.00115\t0.4150\t0.00079\t0.00062
sub-P1536\tAF_R\t0.00118\t0.4200\t0.00081\t0.00064
"""

    # Create file
    file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
    with open(file_path, "w") as f:
        f.write(data)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the metricsinroi module can be imported."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    assert hasattr(metricsinroi, "MultiqcModule")


def test_parse_single_file(reset_multiqc):
    """Test parsing a single metricsinroi file."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create a test file
        header = "sample\troi\tad\tfa\tmd\trd"
        file_content = f"""{header}
sub-P1688\tAC\t0.00127\t0.3245\t0.00093\t0.00076
sub-P1688\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058
sub-P1688\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059
"""

        file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(file_content)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["metricsinroi"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        module = metricsinroi.MultiqcModule()

        # Check that the module parsed the data
        assert module is not None
        assert len(report.general_stats_data) > 0

        # Check that sub-P1688 was parsed with mean FA
        general_stats = list(report.general_stats_data.values())[0]
        assert "sub-P1688" in general_stats
        p1688_row = general_stats["sub-P1688"][0]
        assert "mean_fa" in p1688_row.data
        # Mean FA should be average of 0.3245, 0.4079, 0.4091
        assert abs(p1688_row.data["mean_fa"] - 0.3805) < 0.01

    finally:
        shutil.rmtree(tmpdir)


def test_fa_value_calculation(reset_multiqc):
    """Test mean FA calculation across rois.

    Creates test data where different samples have different FA values
    to verify mean FA calculation.
    """
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create test data with varying FA values
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-HIGH\tROI1\t0.00127\t0.6000\t0.00093\t0.00076
sub-HIGH\tROI2\t0.00109\t0.6500\t0.00075\t0.00058
sub-HIGH\tROI3\t0.00112\t0.7000\t0.00077\t0.00059
sub-MED\tROI1\t0.00130\t0.4000\t0.00095\t0.00078
sub-MED\tROI2\t0.00115\t0.4500\t0.00079\t0.00062
sub-LOW\tROI1\t0.00125\t0.2000\t0.00092\t0.00075
sub-LOW\tROI2\t0.00120\t0.2500\t0.00090\t0.00070
"""

        file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["metricsinroi"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        metricsinroi.MultiqcModule()

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]

        # Check mean FA values
        high_row = general_stats["sub-HIGH"][0]
        assert abs(high_row.data["mean_fa"] - 0.65) < 0.01  # (0.6+0.65+0.7)/3

        med_row = general_stats["sub-MED"][0]
        assert abs(med_row.data["mean_fa"] - 0.425) < 0.01  # (0.4+0.45)/2

        low_row = general_stats["sub-LOW"][0]
        assert abs(low_row.data["mean_fa"] - 0.225) < 0.01  # (0.2+0.25)/2

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_pass(reset_multiqc):
    """Test that samples within IQR range get pass status."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create samples with similar FA values (all within normal range)
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-PASS1\tROI1\t0.00127\t0.4000\t0.00093\t0.00076
sub-PASS1\tROI2\t0.00109\t0.4100\t0.00075\t0.00058
sub-PASS2\tROI1\t0.00112\t0.4050\t0.00077\t0.00059
sub-PASS2\tROI2\t0.00130\t0.4150\t0.00095\t0.00078
sub-PASS3\tROI1\t0.00115\t0.3950\t0.00079\t0.00062
sub-PASS3\tROI2\t0.00118\t0.4200\t0.00081\t0.00064
"""

        file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["metricsinroi"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        module = metricsinroi.MultiqcModule()

        # Check status in one of the sections - all should pass (within IQR)
        assert len(module.sections) > 0
        section = module.sections[0]
        assert '"sub-PASS1": "pass"' in section.status_bar_html
        assert '"sub-PASS2": "pass"' in section.status_bar_html
        assert '"sub-PASS3": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_fail(reset_multiqc):
    """Test that samples outside IQR range get fail status."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create samples where one has extremely different FA (outlier)
        # Need enough samples for IQR to work properly
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-NORMAL1\tROI1\t0.00127\t0.4000\t0.00093\t0.00076
sub-NORMAL1\tROI2\t0.00109\t0.4100\t0.00075\t0.00058
sub-NORMAL2\tROI1\t0.00112\t0.4050\t0.00077\t0.00059
sub-NORMAL2\tROI2\t0.00130\t0.4150\t0.00095\t0.00078
sub-NORMAL3\tROI1\t0.00115\t0.4020\t0.00079\t0.00062
sub-NORMAL3\tROI2\t0.00118\t0.4120\t0.00081\t0.00064
sub-NORMAL4\tROI1\t0.00120\t0.4080\t0.00085\t0.00070
sub-NORMAL4\tROI2\t0.00122\t0.4180\t0.00087\t0.00072
sub-NORMAL5\tROI1\t0.00125\t0.4060\t0.00088\t0.00074
sub-NORMAL5\tROI2\t0.00128\t0.4160\t0.00090\t0.00076
sub-OUTLIER\tROI1\t0.00115\t0.1000\t0.00079\t0.00062
sub-OUTLIER\tROI2\t0.00118\t0.1100\t0.00081\t0.00064
"""

        file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["metricsinroi"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        module = metricsinroi.MultiqcModule()

        # Check status - outlier should fail, normals should pass
        section = module.sections[0]
        assert '"sub-OUTLIER": "fail"' in section.status_bar_html
        assert '"sub-NORMAL1": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-P1688"]
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "rois_mean_stats.tsv")
    report.files["metricsinroi"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "rois_mean_stats",
            "sp_key": "metricsinroi",
        }
    ]

    module = metricsinroi.MultiqcModule()
    assert module is not None

    # Verify that the ignored sample was actually filtered out
    assert len(report.general_stats_data) > 0
    general_stats = list(report.general_stats_data.values())[0]
    assert "sub-P1688" not in general_stats
    assert "sub-P1536" in general_stats

    # Verify only 1 sample remains after filtering
    assert len(general_stats) == 1

    config.sample_names_ignore = []


def test_data_written_to_file(reset_multiqc, test_data_dir):
    """Test that parsed data is written to output file."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "rois_mean_stats.tsv")
    report.files["metricsinroi"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "rois_mean_stats",
            "sp_key": "metricsinroi",
        }
    ]

    module = metricsinroi.MultiqcModule()

    # Check that raw data was saved
    assert module.saved_raw_data is not None
    assert len(module.saved_raw_data) > 0

    # The saved_raw_data should have sample_fa_values and rois
    data_dict = module.saved_raw_data["multiqc_metricsinroi"]
    assert "sample_fa_values" in data_dict
    assert "rois" in data_dict

    # Check that both samples are in sample_fa_values
    assert len(data_dict["sample_fa_values"]) == 2


def test_sections_added(reset_multiqc, test_data_dir):
    """Test that sections with plots are added to the report."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "rois_mean_stats.tsv")
    report.files["metricsinroi"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "rois_mean_stats",
            "sp_key": "metricsinroi",
        }
    ]

    module = metricsinroi.MultiqcModule()

    # Check that sections were added (FA section)
    assert hasattr(module, "sections")
    assert len(module.sections) >= 1

    # Check section names
    section_names = [s.name for s in module.sections]
    assert "Fractional Anisotropy (FA)" in section_names


def test_general_stats_added(reset_multiqc, test_data_dir):
    """Test that general statistics are added to the report."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "rois_mean_stats.tsv")
    report.files["metricsinroi"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "rois_mean_stats",
            "sp_key": "metricsinroi",
        }
    ]

    metricsinroi.MultiqcModule()

    # Check that general stats data was added to the report
    assert len(report.general_stats_data) > 0

    # Check that both samples have mean_fa field
    general_stats = list(report.general_stats_data.values())[0]
    assert len(general_stats) == 2

    for sample_name in ["sub-P1688", "sub-P1536"]:
        assert sample_name in general_stats
        sample_row = general_stats[sample_name][0]
        assert "mean_fa" in sample_row.data


def test_configurable_thresholds(reset_multiqc, test_data_dir):
    """Test that custom IQR multiplier can be configured."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    # Set custom IQR multiplier
    config.metricsinroi = {"iqr_multiplier": 2}
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "rois_mean_stats.tsv")
    report.files["metricsinroi"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "rois_mean_stats",
            "sp_key": "metricsinroi",
        }
    ]

    module = metricsinroi.MultiqcModule()

    # Module should work with custom IQR multiplier
    assert module is not None
    assert len(module.sections) > 0

    # Cleanup config
    delattr(config, "metricsinroi")


def test_single_sample_handling(reset_multiqc):
    """Test that the module handles single-sample files correctly."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create single-sample file
        header = "sample\troi\tad\tfa\tmd\trd"
        single_data = f"""{header}
sub-SINGLE\tROI1\t0.00127\t0.3245\t0.00093\t0.00076
sub-SINGLE\tROI2\t0.00109\t0.4079\t0.00075\t0.00058
sub-SINGLE\tROI3\t0.00112\t0.4091\t0.00077\t0.00059
"""

        file_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(single_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}
        config.preserve_module_raw_data = True

        report.files["metricsinroi"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        # Module should not crash with single sample
        module = metricsinroi.MultiqcModule()
        assert module is not None

        # Check that sections were added (FA section)
        assert len(module.sections) >= 1

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]
        assert "sub-SINGLE" in general_stats

        # Single sample should have mean FA
        single_row = general_stats["sub-SINGLE"][0]
        assert "mean_fa" in single_row.data

    finally:
        shutil.rmtree(tmpdir)


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "rois_mean_stats.tsv")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["metricsinroi"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "rois_mean_stats",
                "sp_key": "metricsinroi",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            metricsinroi.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
