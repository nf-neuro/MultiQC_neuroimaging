"""
Tests for the tractometry module.
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
    if "tractometry" not in config.sp:
        config.update_dict(config.sp, {"tractometry":
                                       {"fn": "bundles_mean_stats.tsv"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Sample tractometry data
    header = (
        "sample\tsession\tbundle\tad\tfa\tmd\trd\t" +
        "avg_length\tstreamlines_count"
    )
    data = f"""{header}
sub-P1688\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-P1688\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-P1688\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-P1536\t\tAC\t0.00130\t0.3350\t0.00095\t0.00078\t68.120\t8
sub-P1536\t\tAF_L\t0.00115\t0.4150\t0.00079\t0.00062\t112.450\t98
sub-P1536\t\tAF_R\t0.00118\t0.4200\t0.00081\t0.00064\t115.320\t320
sub-P1536\t\tvolume\t0.0\t0.0\t0.0\t0.0\t0.0\t0
"""

    # Create file
    file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
    with open(file_path, "w") as f:
        f.write(data)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the tractometry module can be imported."""
    from neuroimagingQC.modules.tractometry import tractometry

    assert hasattr(tractometry, "MultiqcModule")


def test_parse_single_file(reset_multiqc):
    """Test parsing a single tractometry file."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # Create a test file
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        file_content = f"""{header}
sub-P1688\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-P1688\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-P1688\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(file_content)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        module = tractometry.MultiqcModule()

        # Check that the module parsed the data
        assert module is not None
        assert len(report.general_stats_data) > 0

        # Check that sub-P1688 was parsed with 3 bundles
        general_stats = list(report.general_stats_data.values())[0]
        assert "sub-P1688" in general_stats
        p1688_row = general_stats["sub-P1688"][0]
        assert "bundle_percentage" in p1688_row.data
        # 3 bundles detected out of 3 total = 100%
        assert p1688_row.data["bundle_percentage"] == 100.0

    finally:
        shutil.rmtree(tmpdir)


def test_bundle_percentage_calculation(reset_multiqc):
    """Test bundle percentage calculation with partial extraction.

    Creates test data where different samples have different numbers
    of bundles extracted to verify percentage calculation.
    """
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # Create test data
        # Total bundles: AC, AF_L, AF_R (3 bundles)
        # sub-FULL: has all 3 bundles = 100%
        # sub-PARTIAL: has 2 bundles = 66.7%
        # sub-POOR: has 1 bundle = 33.3%
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        test_data = f"""{header}
sub-FULL\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-PARTIAL\t\tAC\t0.00130\t0.3350\t0.00095\t0.00078\t68.120\t8
sub-PARTIAL\t\tAF_L\t0.00115\t0.4150\t0.00079\t0.00062\t112.450\t98
sub-POOR\t\tAC\t0.00125\t0.3200\t0.00092\t0.00075\t64.500\t5
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        tractometry.MultiqcModule()

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]

        # Check bundle percentages
        full_row = general_stats["sub-FULL"][0]
        assert full_row.data["bundle_percentage"] == 100.0

        partial_row = general_stats["sub-PARTIAL"][0]
        assert abs(partial_row.data["bundle_percentage"] - 66.67) < 0.1

        poor_row = general_stats["sub-POOR"][0]
        assert abs(poor_row.data["bundle_percentage"] - 33.33) < 0.1

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_pass(reset_multiqc):
    """Test that samples with >=90% bundles get pass status."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # Sample with all bundles should pass
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        test_data = f"""{header}
sub-PASS\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-PASS\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-PASS\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        module = tractometry.MultiqcModule()

        # Check status in one of the sections
        assert len(module.sections) > 0
        section = module.sections[0]
        assert '"sub-PASS": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_warn(reset_multiqc):
    """Test that samples with 80-90% bundles get warn status."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # Create data with 9 total bundles
        # sub-WARN will have 8 bundles = 88.89% (should warn)
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        test_data = f"""{header}
sub-WARN\t\tB1\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-WARN\t\tB2\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-WARN\t\tB3\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-WARN\t\tB4\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-WARN\t\tB5\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-WARN\t\tB6\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-WARN\t\tB7\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-WARN\t\tB8\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tB1\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tB2\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tB3\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-FULL\t\tB4\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tB5\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tB6\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
sub-FULL\t\tB7\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tB8\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tB9\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        module = tractometry.MultiqcModule()

        # Check status
        section = module.sections[0]
        assert '"sub-WARN": "warn"' in section.status_bar_html
        assert '"sub-FULL": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_status_assignment_fail(reset_multiqc):
    """Test that samples with <80% bundles get fail status."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # sub-FAIL will have 1 out of 3 bundles = 33.3% (should fail)
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        test_data = f"""{header}
sub-FAIL\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-FULL\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-FULL\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        module = tractometry.MultiqcModule()

        # Check status
        section = module.sections[0]
        assert '"sub-FAIL": "fail"' in section.status_bar_html
        assert '"sub-FULL": "pass"' in section.status_bar_html

    finally:
        shutil.rmtree(tmpdir)


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimagingQC.modules.tractometry import tractometry

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-P1688"]
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "bundles_mean_stats.tsv")
    report.files["tractometry"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "bundles_mean_stats",
            "sp_key": "tractometry",
        }
    ]

    module = tractometry.MultiqcModule()
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
    from neuroimagingQC.modules.tractometry import tractometry

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "bundles_mean_stats.tsv")
    report.files["tractometry"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "bundles_mean_stats",
            "sp_key": "tractometry",
        }
    ]

    module = tractometry.MultiqcModule()

    # Check that raw data was saved
    assert module.saved_raw_data is not None
    assert len(module.saved_raw_data) > 0

    # The saved_raw_data should have sample_counts and bundles
    data_dict = module.saved_raw_data["multiqc_tractometry"]
    assert "sample_counts" in data_dict
    assert "bundles" in data_dict

    # Check that both samples are in sample_counts
    assert len(data_dict["sample_counts"]) == 2


def test_sections_added(reset_multiqc, test_data_dir):
    """Test that sections with plots are added to the report."""
    from neuroimagingQC.modules.tractometry import tractometry

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "bundles_mean_stats.tsv")
    report.files["tractometry"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "bundles_mean_stats",
            "sp_key": "tractometry",
        }
    ]

    module = tractometry.MultiqcModule()

    # Check that sections were added (one for each metric with data)
    # Test data has fa, volume, and streamlines_count
    assert hasattr(module, "sections")
    # May have 2-3 sections depending on which metrics have data
    assert len(module.sections) >= 2

    # Check section names
    section_names = [s.name for s in module.sections]
    # At minimum, FA and streamlines should be present
    assert (
        "Fractional Anisotropy (FA)" in section_names
        or "Streamline Count" in section_names
    )


def test_general_stats_added(reset_multiqc, test_data_dir):
    """Test that general statistics are added to the report."""
    from neuroimagingQC.modules.tractometry import tractometry

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "bundles_mean_stats.tsv")
    report.files["tractometry"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "bundles_mean_stats",
            "sp_key": "tractometry",
        }
    ]

    tractometry.MultiqcModule()

    # Check that general stats data was added to the report
    assert len(report.general_stats_data) > 0

    # Check that both samples have bundle_percentage field
    general_stats = list(report.general_stats_data.values())[0]
    assert len(general_stats) == 2

    for sample_name in ["sub-P1688", "sub-P1536"]:
        assert sample_name in general_stats
        sample_row = general_stats[sample_name][0]
        assert "bundle_percentage" in sample_row.data


def test_configurable_thresholds(reset_multiqc, test_data_dir):
    """Test that custom thresholds can be configured."""
    from neuroimagingQC.modules.tractometry import tractometry

    # Set custom thresholds
    config.tractometry = {"warn_threshold": 95, "fail_threshold": 85}
    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "bundles_mean_stats.tsv")
    report.files["tractometry"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "bundles_mean_stats",
            "sp_key": "tractometry",
        }
    ]

    module = tractometry.MultiqcModule()

    # Module should work with custom thresholds
    assert module is not None
    assert len(module.sections) > 0

    # Cleanup config
    delattr(config, "tractometry")


def test_single_sample_handling(reset_multiqc):
    """Test that the module handles single-sample files correctly."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        # Create single-sample file
        header = (
            "sample\tsession\tbundle\tad\tfa\tmd\trd\t"
            + "avg_length\tstreamlines_count"
        )
        single_data = f"""{header}
sub-SINGLE\t\tAC\t0.00127\t0.3245\t0.00093\t0.00076\t65.833\t6
sub-SINGLE\t\tAF_L\t0.00109\t0.4079\t0.00075\t0.00058\t114.139\t105
sub-SINGLE\t\tAF_R\t0.00112\t0.4091\t0.00077\t0.00059\t116.825\t340
"""

        file_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(file_path, "w") as f:
            f.write(single_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}
        config.preserve_module_raw_data = True

        report.files["tractometry"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        # Module should not crash with single sample
        module = tractometry.MultiqcModule()
        assert module is not None

        # Check that sections were added (FA and streamlines at minimum)
        assert len(module.sections) >= 2

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]
        assert "sub-SINGLE" in general_stats

        # Single sample with all bundles should have 100%
        single_row = general_stats["sub-SINGLE"][0]
        assert single_row.data["bundle_percentage"] == 100.0

    finally:
        shutil.rmtree(tmpdir)


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimagingQC.modules.tractometry import tractometry

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "bundles_mean_stats.tsv")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["tractometry"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "bundles_mean_stats",
                "sp_key": "tractometry",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            tractometry.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
