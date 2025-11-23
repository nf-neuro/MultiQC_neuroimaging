"""
Tests for the subcortical module.
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
    if "subcortical/volume" not in config.sp:
        config.update_dict(
            config.sp, {"subcortical/volume":
                        {"fn": "*_subcortical_volumes.tsv"}}
        )
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Sample subcortical data
    data = """Sample\tmAmyg_L\tmAmyg_R\tlAmyg_L\tlAmyg_R\tGP_L\tGP_R
sub-P0933\t1010.4\t1362.1\t418.4\t588.4\t1935.1\t2199.6
sub-P1569\t894.7\t1213.4\t505.8\t685.4\t2108.5\t2225.3
sub-P0201\t920.8\t1295.0\t380.7\t548.6\t1866.4\t2063.2
sub-P1028\t992.0\t1419.6\t466.5\t510.0\t2087.4\t2113.4
sub-P0190\t1002.7\t1307.1\t476.0\t680.7\t2131.9\t2259.5
"""

    # Create file
    file_path = os.path.join(tmpdir, "Test_subcortical_volumes.tsv")
    with open(file_path, "w") as f:
        f.write(data)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the subcortical module can be imported."""
    from neuroimagingQC.modules.subcortical import subcortical

    assert hasattr(subcortical, "MultiqcModule")


def test_parse_single_file(reset_multiqc):
    """Test parsing a single subcortical volume file."""
    from neuroimagingQC.modules.subcortical import subcortical

    # Create a mock file object
    file_content = """Sample\tmAmyg_L\tmAmyg_R\tlAmyg_L\tlAmyg_R\tGP_L\tGP_R
sub-P0933\t1010.4\t1362.1\t418.4\t588.4\t1935.1\t2199.6
sub-P1569\t894.7\t1213.4\t505.8\t685.4\t2108.5\t2225.3
sub-P0201\t920.8\t1295.0\t380.7\t548.6\t1866.4\t2063.2"""

    f = {
        "f": file_content,
        "fn": "Test_subcortical_volumes.tsv",
        "s_name": "Test",
    }

    # Create a minimal module instance just to access the parse method
    module = object.__new__(subcortical.MultiqcModule)
    result = module.parse_subcortical_file(f)

    # Check that all samples were parsed
    assert len(result) == 3
    assert "sub-P0933" in result
    assert "sub-P1569" in result
    assert "sub-P0201" in result

    # Check that all regions were parsed
    assert len(result["sub-P0933"]) == 6
    assert "mAmyg_L" in result["sub-P0933"]
    assert "mAmyg_R" in result["sub-P0933"]
    assert "GP_L" in result["sub-P0933"]
    assert "GP_R" in result["sub-P0933"]

    # Check specific values
    assert result["sub-P0933"]["mAmyg_L"] == 1010.4
    assert result["sub-P0933"]["mAmyg_R"] == 1362.1
    assert result["sub-P0933"]["GP_L"] == 1935.1


def test_iqr_calculation(reset_multiqc):
    """Test IQR-based outlier detection with known outlier.

    Creates test data where one sample has a clear outlier region
    and verifies the module correctly calculates outlier percentage.
    """
    from neuroimagingQC.modules.subcortical import subcortical

    tmpdir = tempfile.mkdtemp()

    try:
        # Create test data with known outliers
        # Region 1: values [100, 200, 300, 400, 500] - no outliers
        # Region 2: values [100, 100, 100, 100, 5000] - outlier at 5000
        # Sample5 should have 50% outliers (1 out of 2 regions)
        test_data = """Sample\tregion1\tregion2
sub-sample1\t100.0\t100.0
sub-sample2\t200.0\t100.0
sub-sample3\t300.0\t100.0
sub-sample4\t400.0\t100.0
sub-sample5\t500.0\t5000.0
"""

        file_path = os.path.join(tmpdir, "Test_subcortical_volumes.tsv")
        with open(file_path, "w") as f:
            f.write(test_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["subcortical/volume"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "Test",
                "sp_key": "subcortical/volume",
            }
        ]

        subcortical.MultiqcModule()

        # Check that general stats were added
        assert len(report.general_stats_data) > 0
        general_stats = list(report.general_stats_data.values())[0]

        # Check that sample5 has 50% outliers
        sample5_row = general_stats["sub-sample5"][0]
        assert "region_pct" in sample5_row.data
        assert sample5_row.data["region_pct"] == 50.0

        # Other samples should have 0% outliers
        for sample in ["sub-sample1", "sub-sample2", "sub-sample3",
                       "sub-sample4"]:
            sample_row = general_stats[sample][0]
            assert sample_row.data["region_pct"] == 0.0

    finally:
        shutil.rmtree(tmpdir)


def test_subcortical_files(reset_multiqc, test_data_dir):
    """Test parsing subcortical TSV files."""
    from neuroimagingQC.modules.subcortical import subcortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "Test_subcortical_volumes.tsv")
    report.files["subcortical/volume"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "subcortical/volume",
        }
    ]

    module = subcortical.MultiqcModule()
    assert module is not None


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimagingQC.modules.subcortical import subcortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-P0933"]
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "Test_subcortical_volumes.tsv")
    report.files["subcortical/volume"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "subcortical/volume",
        }
    ]

    module = subcortical.MultiqcModule()
    assert module is not None

    # Verify that the ignored sample was actually filtered out
    data_dict = module.saved_raw_data["multiqc_subcortical_data"]
    assert "sub-P0933" not in data_dict, "Ignored sample should not be " + \
        "in output"

    # Verify that other samples are still present
    assert len(data_dict) == 4, (
        f"Expected 4 samples after filtering, " f"got {len(data_dict)}"
    )

    config.sample_names_ignore = []


def test_data_written_to_file(reset_multiqc, test_data_dir):
    """Test that parsed data is written to output file."""
    from neuroimagingQC.modules.subcortical import subcortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    file_path = os.path.join(test_data_dir, "Test_subcortical_volumes.tsv")
    report.files["subcortical/volume"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "subcortical/volume",
        }
    ]

    module = subcortical.MultiqcModule()

    # Check that raw data was saved
    assert module.saved_raw_data is not None
    assert len(module.saved_raw_data) > 0

    # The saved_raw_data has one key 'multiqc_subcortical_data'
    # and its value is the actual data dictionary with samples
    data_dict = module.saved_raw_data["multiqc_subcortical_data"]
    assert len(data_dict) == 5  # 5 samples in test data


def test_section_added(reset_multiqc, test_data_dir):
    """Test that a section with plot is added to the report."""
    from neuroimagingQC.modules.subcortical import subcortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "Test_subcortical_volumes.tsv")
    report.files["subcortical/volume"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "subcortical/volume",
        }
    ]

    module = subcortical.MultiqcModule()

    # Check that sections were added
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check section properties
    section = module.sections[0]
    assert section.name == "Subcortical Volume Distribution"
    assert section.anchor == "subcortical_volumes"
    assert hasattr(section, "plot")


def test_general_stats_added(reset_multiqc, test_data_dir):
    """Test that general statistics are added to the report."""
    from neuroimagingQC.modules.subcortical import subcortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    file_path = os.path.join(test_data_dir, "Test_subcortical_volumes.tsv")
    report.files["subcortical/volume"] = [
        {
            "fn": file_path,
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "subcortical/volume",
        }
    ]

    module = subcortical.MultiqcModule()

    # Check that sections were added (which include the plots)
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check that general stats data was added to the report
    assert len(report.general_stats_data) > 0

    # Check that all samples have the region_pct field
    general_stats = list(report.general_stats_data.values())[0]
    assert len(general_stats) == 5  # 5 samples in test data

    for sample_name in [
        "sub-P0933",
        "sub-P1569",
        "sub-P0201",
        "sub-P1028",
        "sub-P0190",
    ]:
        assert sample_name in general_stats
        sample_row = general_stats[sample_name][0]
        assert "region_pct" in sample_row.data


def test_single_sample_handling(reset_multiqc):
    """Test that the module handles single-sample files correctly.

    When there's only one sample, IQR calculation cannot determine
    outliers, so the module should handle this gracefully.
    """
    from neuroimagingQC.modules.subcortical import subcortical

    tmpdir = tempfile.mkdtemp()

    try:
        # Create single-sample file
        single_data = """Sample\tmAmyg_L\tmAmyg_R\tlAmyg_L\tlAmyg_R\tGP_L\tGP_R
sub-SINGLE\t1010.4\t1362.1\t418.4\t588.4\t1935.1\t2199.6
"""

        file_path = os.path.join(tmpdir, "Test_subcortical_volumes.tsv")
        with open(file_path, "w") as f:
            f.write(single_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}
        config.preserve_module_raw_data = True

        report.files["subcortical/volume"] = [
            {
                "fn": file_path,
                "root": tmpdir,
                "s_name": "Test",
                "sp_key": "subcortical/volume",
            }
        ]

        # Module should not crash with single sample
        module = subcortical.MultiqcModule()
        assert module is not None

        # Check that the sample was parsed
        data_dict = module.saved_raw_data["multiqc_subcortical_data"]
        assert len(data_dict) == 1
        assert "sub-SINGLE" in data_dict

        # Check that sections were added
        assert len(module.sections) > 0

        # Check that general stats were added
        assert len(report.general_stats_data) > 0

        # With a single sample, outlier percentage should be 0.0
        # (no basis for comparison)
        module_data = list(report.general_stats_data.values())[0]
        single_row = module_data["sub-SINGLE"][0]
        assert "region_pct" in single_row.data
        assert single_row.data["region_pct"] == 0.0

    finally:
        shutil.rmtree(tmpdir)


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimagingQC.modules.subcortical import subcortical

    tmpdir = tempfile.mkdtemp()

    try:
        empty_path = os.path.join(tmpdir, "Test_subcortical_volumes.tsv")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["subcortical/volume"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "Test",
                "sp_key": "subcortical/volume",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            subcortical.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)


def test_malformed_file_handling(reset_multiqc):
    """Test handling of malformed subcortical files."""
    from neuroimagingQC.modules.subcortical import subcortical

    tmpdir = tempfile.mkdtemp()

    try:
        # Create file with header only (no data rows)
        bad_path = os.path.join(tmpdir, "Test_subcortical_volumes.tsv")
        with open(bad_path, "w") as f:
            f.write("Sample\tmAmyg_L\tmAmyg_R\n")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["subcortical/volume"] = [
            {
                "fn": bad_path,
                "root": tmpdir,
                "s_name": "Test",
                "sp_key": "subcortical/volume",
            }
        ]

        # Module should raise exception for file with no data
        with pytest.raises(ModuleNoSamplesFound):
            subcortical.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)
