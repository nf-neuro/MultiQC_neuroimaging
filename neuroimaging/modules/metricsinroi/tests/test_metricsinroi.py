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
        # Module should have sections with plots
        assert len(module.sections) > 0

    finally:
        shutil.rmtree(tmpdir)


def test_per_roi_violin_plot(reset_multiqc):
    """Test that violin plots are created per ROI.

    Creates test data where different samples have different FA values
    to verify per-ROI violin plot creation.
    """
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create test data with varying FA values across different ROIs
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-001\tROI1\t0.00127\t0.6000\t0.00093\t0.00076
sub-001\tROI2\t0.00109\t0.5000\t0.00075\t0.00058
sub-001\tROI3\t0.00112\t0.4000\t0.00077\t0.00059
sub-002\tROI1\t0.00130\t0.6100\t0.00095\t0.00078
sub-002\tROI2\t0.00115\t0.5100\t0.00079\t0.00062
sub-002\tROI3\t0.00125\t0.4100\t0.00092\t0.00075
sub-003\tROI1\t0.00120\t0.5900\t0.00090\t0.00070
sub-003\tROI2\t0.00118\t0.4900\t0.00088\t0.00068
sub-003\tROI3\t0.00122\t0.3900\t0.00089\t0.00069
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

        # Check that sections with violin plots were created
        assert len(module.sections) > 0
        # Should have FA section with violin plot containing all 3 ROIs
        assert any(s.name == "Fractional Anisotropy (FA)" for s in module.sections)

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_pass(reset_multiqc):
    """Test that samples within IQR range for all ROIs get pass status."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create samples with similar FA values within normal range for each ROI
        # Different ROIs have different FA ranges to test per-ROI detection
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-PASS1\tROI1\t0.00127\t0.4000\t0.00093\t0.00076
sub-PASS1\tROI2\t0.00109\t0.5000\t0.00075\t0.00058
sub-PASS2\tROI1\t0.00112\t0.4050\t0.00077\t0.00059
sub-PASS2\tROI2\t0.00130\t0.5100\t0.00095\t0.00078
sub-PASS3\tROI1\t0.00115\t0.3950\t0.00079\t0.00062
sub-PASS3\tROI2\t0.00118\t0.5050\t0.00081\t0.00064
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

        # Check status in one of the sections - all should pass (within IQR for all ROIs)
        # ROI1 has FA ~0.4, ROI2 has FA ~0.5, all samples are within range for both
        assert len(module.sections) > 0
        section = module.sections[0]
        assert '"sub-PASS1": "pass"' in section.status_bar_html
        assert '"sub-PASS2": "pass"' in section.status_bar_html
        assert '"sub-PASS3": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_fail(reset_multiqc):
    """Test that samples outside IQR range in ANY ROI get fail status."""
    from neuroimaging.modules.metricsinroi import metricsinroi

    tmpdir = tempfile.mkdtemp()

    try:
        # Create samples where one has extremely different FA in one specific ROI (outlier)
        # This tests that per-ROI IQR detection works - sample fails if outlier in ANY ROI
        # Need enough samples for IQR to work properly
        header = "sample\troi\tad\tfa\tmd\trd"
        test_data = f"""{header}
sub-NORMAL1\tROI1\t0.00127\t0.4000\t0.00093\t0.00076
sub-NORMAL1\tROI2\t0.00109\t0.5000\t0.00075\t0.00058
sub-NORMAL2\tROI1\t0.00112\t0.4050\t0.00077\t0.00059
sub-NORMAL2\tROI2\t0.00130\t0.5100\t0.00095\t0.00078
sub-NORMAL3\tROI1\t0.00115\t0.4020\t0.00079\t0.00062
sub-NORMAL3\tROI2\t0.00118\t0.5050\t0.00081\t0.00064
sub-NORMAL4\tROI1\t0.00120\t0.4080\t0.00085\t0.00070
sub-NORMAL4\tROI2\t0.00122\t0.5150\t0.00087\t0.00072
sub-NORMAL5\tROI1\t0.00125\t0.4060\t0.00088\t0.00074
sub-NORMAL5\tROI2\t0.00128\t0.5080\t0.00090\t0.00076
sub-OUTLIER\tROI1\t0.00115\t0.1000\t0.00079\t0.00062
sub-OUTLIER\tROI2\t0.00118\t0.5120\t0.00081\t0.00064
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

        # Check status - outlier should fail even though it's only an outlier in ROI1
        # (ROI2 value is normal), normals should pass
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

    # Verify that the ignored sample was actually filtered out from saved data
    assert len(module.saved_raw_data) > 0
    data_dict = module.saved_raw_data["multiqc_metricsinroi"]
    
    # Extract samples from the rois data structure
    samples = set()
    for roi_name, roi_data in data_dict["rois"].items():
        samples.update(roi_data.keys())
    
    # Check that sub-P1688 was ignored
    assert "sub-P1688" not in samples
    assert "sub-P1536" in samples

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

    # The saved_raw_data should have rois
    data_dict = module.saved_raw_data["multiqc_metricsinroi"]
    assert "rois" in data_dict

    # Check that ROIs contain sample data
    assert len(data_dict["rois"]) > 0


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


def test_sections_with_violin_plots(reset_multiqc, test_data_dir):
    """Test that sections with violin plots are created."""
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

    # Check that sections were created with violin plots
    assert len(module.sections) > 0
    
    # Check that FA section exists
    section_names = [s.name for s in module.sections]
    assert "Fractional Anisotropy (FA)" in section_names


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
        
        # Check that FA section exists
        section_names = [s.name for s in module.sections]
        assert "Fractional Anisotropy (FA)" in section_names

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
