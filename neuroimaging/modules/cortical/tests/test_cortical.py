"""
Tests for the cortical module.
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
    if "cortical/volume" not in config.sp:
        config.update_dict(config.sp, {"cortical/volume": {"fn": "cortical_*_volume_*.tsv"}})
    yield
    config.reset()
    report.reset()


@pytest.fixture
def test_data_dir():
    """Create a temporary directory with test data files."""
    tmpdir = tempfile.mkdtemp()

    # Sample left hemisphere data
    lh_data = """Sample\tlh_SFG_L_6_1\tlh_SFG_L_6_2\tlh_MFG_L_7_1
sub-P1356\t2972.0\t2489.0\t3304.0
sub-P1536\t2040.0\t1648.0\t2858.0
sub-P1107\t1780.0\t2057.0\t3716.0
sub-P0959\t1453.0\t2316.0\t4164.0
sub-P0838\t1940.0\t2673.0\t4261.0
"""

    # Sample right hemisphere data
    rh_data = """Sample\trh_SFG_R_6_1\trh_SFG_R_6_2\trh_MFG_R_7_1
sub-P1356\t2850.0\t2300.0\t3100.0
sub-P1536\t2100.0\t1700.0\t2900.0
sub-P1107\t1800.0\t2100.0\t3700.0
sub-P0959\t1500.0\t2400.0\t4200.0
sub-P0838\t2000.0\t2700.0\t4300.0
"""

    # Create files
    lh_path = os.path.join(tmpdir, "cortical_Test_volume_lh.tsv")
    with open(lh_path, "w") as f:
        f.write(lh_data)

    rh_path = os.path.join(tmpdir, "cortical_Test_volume_rh.tsv")
    with open(rh_path, "w") as f:
        f.write(rh_data)

    yield tmpdir

    shutil.rmtree(tmpdir)


def test_module_import():
    """Test that the cortical module can be imported."""
    from neuroimaging.modules.cortical import cortical

    assert hasattr(cortical, "MultiqcModule")


def test_parse_single_file(reset_multiqc):
    """Test parsing a single cortical volume file."""
    from neuroimaging.modules.cortical import cortical

    # Create a mock file object
    file_content = """Sample\tlh_SFG_L_6_1\tlh_SFG_L_6_2\tlh_MFG_L_7_1
sub-P1356\t2972.0\t2489.0\t3304.0
sub-P1536\t2040.0\t1648.0\t2858.0
sub-P1107\t1780.0\t2057.0\t3716.0"""

    f = {
        "f": file_content,
        "fn": "cortical_Test_volume_lh.tsv",
        "s_name": "Test",
    }

    # Create a minimal module instance just to access the parse method
    # We can't call __init__ so we manually instantiate
    module = object.__new__(cortical.MultiqcModule)
    result = module.parse_cortical_file(f)

    # Check that all samples were parsed
    assert len(result) == 3
    assert "sub-P1356" in result
    assert "sub-P1536" in result
    assert "sub-P1107" in result

    # Check that all regions were parsed
    assert len(result["sub-P1356"]) == 3
    assert "lh_SFG_L_6_1" in result["sub-P1356"]
    assert "lh_SFG_L_6_2" in result["sub-P1356"]
    assert "lh_MFG_L_7_1" in result["sub-P1356"]

    # Check specific values
    assert result["sub-P1356"]["lh_SFG_L_6_1"] == 2972.0
    assert result["sub-P1356"]["lh_SFG_L_6_2"] == 2489.0
    assert result["sub-P1356"]["lh_MFG_L_7_1"] == 3304.0


def test_iqr_calculation(reset_multiqc):
    """Test IQR-based outlier detection."""
    from neuroimaging.modules.cortical import cortical

    # Create test data with known outliers
    # Region 1: values [100, 200, 300, 400, 500] - no outliers expected
    # Region 2: values [100, 100, 100, 100, 1000] - outlier at 1000
    cortical_data = {
        "sample1": {"region1": 100.0, "region2": 100.0},
        "sample2": {"region1": 200.0, "region2": 100.0},
        "sample3": {"region1": 300.0, "region2": 100.0},
        "sample4": {"region1": 400.0, "region2": 100.0},
        "sample5": {"region1": 500.0, "region2": 1000.0},
    }

    # Create a minimal module instance just to access the method
    module = object.__new__(cortical.MultiqcModule)
    percentages = module._calculate_outlier_percentages(cortical_data, 3)

    # Check that sample5 is identified as having outliers
    # It should have 50% outliers (1 out of 2 regions)
    assert percentages["sample5"] == 50.0

    # Other samples should have no outliers or fewer
    assert percentages["sample1"] == 0.0
    assert percentages["sample2"] == 0.0
    assert percentages["sample3"] == 0.0
    assert percentages["sample4"] == 0.0


def test_iqr_bounds_calculation(reset_multiqc):
    """Test that IQR bounds are correctly calculated."""
    from neuroimaging.modules.cortical import cortical

    # Create test data with known distribution
    # Q1 = 200, Q3 = 400, IQR = 200
    # Lower bound = 200 - 3*200 = -400
    # Upper bound = 400 + 3*200 = 1000
    cortical_data = {
        "sample1": {"region1": 100.0},
        "sample2": {"region1": 200.0},
        "sample3": {"region1": 300.0},
        "sample4": {"region1": 400.0},
        "sample5": {"region1": 500.0},
    }

    # Create a minimal module instance just to access the method
    module = object.__new__(cortical.MultiqcModule)
    percentages = module._calculate_outlier_percentages(cortical_data, 3)

    # All values should be within bounds
    for sample, pct in percentages.items():
        assert pct == 0.0


def test_cortical_files(reset_multiqc, test_data_dir):
    """Test parsing cortical TSV files."""
    from neuroimaging.modules.cortical import cortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["cortical/volume"] = [
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_lh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_rh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
    ]

    module = cortical.MultiqcModule()
    assert module is not None


def test_ignore_samples(reset_multiqc, test_data_dir):
    """Test ignore_samples configuration."""
    from neuroimaging.modules.cortical import cortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.sample_names_ignore = ["sub-P1356"]
    config.preserve_module_raw_data = True

    report.files["cortical/volume"] = [
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_lh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_rh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
    ]

    module = cortical.MultiqcModule()
    assert module is not None

    # Verify that the ignored sample was actually filtered out
    data_dict = module.saved_raw_data["multiqc_cortical_data"]
    assert "sub-P1356" not in data_dict, "Ignored sample should not be in output"

    # Verify that other samples are still present
    assert len(data_dict) == 4, f"Expected 4 samples after filtering, got {len(data_dict)}"

    config.sample_names_ignore = []


def test_empty_file_handling(reset_multiqc):
    """Test handling of empty files."""
    from neuroimaging.modules.cortical import cortical

    tmpdir = tempfile.mkdtemp()

    try:
        # Create empty file
        empty_path = os.path.join(tmpdir, "cortical_Empty_volume_lh.tsv")
        with open(empty_path, "w") as f:
            f.write("")

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}

        report.files["cortical/volume"] = [
            {
                "fn": empty_path,
                "root": tmpdir,
                "s_name": "Empty",
                "sp_key": "cortical/volume",
            }
        ]

        with pytest.raises(ModuleNoSamplesFound):
            cortical.MultiqcModule()
    finally:
        shutil.rmtree(tmpdir)


def test_data_written_to_file(reset_multiqc, test_data_dir):
    """Test that parsed data is written to output file."""
    from neuroimaging.modules.cortical import cortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    report.files["cortical/volume"] = [
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_lh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_rh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
    ]

    module = cortical.MultiqcModule()

    # Check that raw data was saved
    assert module.saved_raw_data is not None
    assert len(module.saved_raw_data) > 0

    # The saved_raw_data has one key which is the filename
    # 'multiqc_cortical_data'
    # and its value is the actual data dictionary with samples
    data_dict = module.saved_raw_data["multiqc_cortical_data"]
    assert len(data_dict) == 5  # 5 samples in test data


def test_hemisphere_merging(reset_multiqc, test_data_dir):
    """Test that left and right hemisphere data are merged correctly."""
    from neuroimaging.modules.cortical import cortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}
    config.preserve_module_raw_data = True

    report.files["cortical/volume"] = [
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_lh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_rh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
    ]

    module = cortical.MultiqcModule()

    # Get the data dictionary
    data_dict = module.saved_raw_data["multiqc_cortical_data"]

    # Get first sample's data
    first_sample = list(data_dict.keys())[0]
    regions = data_dict[first_sample]

    # Check that both hemispheres are present
    has_lh = any(r.startswith("lh_") for r in regions.keys())
    has_rh = any(r.startswith("rh_") for r in regions.keys())

    assert has_lh, "Left hemisphere regions should be present"
    assert has_rh, "Right hemisphere regions should be present"


def test_general_stats_added(reset_multiqc, test_data_dir):
    """Test that general statistics are added to the report."""
    from neuroimaging.modules.cortical import cortical

    config.analysis_dir = [test_data_dir]
    config.kwargs = {"single_subject": False}

    report.files["cortical/volume"] = [
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_lh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
        {
            "fn": os.path.join(test_data_dir, "cortical_Test_volume_rh.tsv"),
            "root": test_data_dir,
            "s_name": "Test",
            "sp_key": "cortical/volume",
        },
    ]

    module = cortical.MultiqcModule()

    # Check that sections were added (which include the plots)
    assert hasattr(module, "sections")
    assert len(module.sections) > 0

    # Check that general stats data was added to the report
    assert len(report.general_stats_data) > 0


def test_single_sample_handling(reset_multiqc):
    """Test that the module handles single-sample files correctly.

    When there's only one sample, IQR calculation cannot determine
    outliers, so the module should handle this gracefully.
    """
    from neuroimaging.modules.cortical import cortical

    tmpdir = tempfile.mkdtemp()

    try:
        # Create single-sample files
        lh_data = """Sample\tlh_SFG_L_6_1\tlh_SFG_L_6_2\tlh_MFG_L_7_1
sub-SINGLE\t2972.0\t2489.0\t3304.0
"""

        rh_data = """Sample\trh_SFG_R_6_1\trh_SFG_R_6_2\trh_MFG_R_7_1
sub-SINGLE\t2850.0\t2300.0\t3100.0
"""

        lh_path = os.path.join(tmpdir, "cortical_Single_volume_lh.tsv")
        with open(lh_path, "w") as f:
            f.write(lh_data)

        rh_path = os.path.join(tmpdir, "cortical_Single_volume_rh.tsv")
        with open(rh_path, "w") as f:
            f.write(rh_data)

        config.analysis_dir = [tmpdir]
        config.kwargs = {"single_subject": False}
        config.preserve_module_raw_data = True

        report.files["cortical/volume"] = [
            {
                "fn": lh_path,
                "root": tmpdir,
                "s_name": "Single",
                "sp_key": "cortical/volume",
            },
            {
                "fn": rh_path,
                "root": tmpdir,
                "s_name": "Single",
                "sp_key": "cortical/volume",
            },
        ]

        # Module should not crash with single sample
        module = cortical.MultiqcModule()
        assert module is not None

        # Check that the sample was parsed
        data_dict = module.saved_raw_data["multiqc_cortical_data"]
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
