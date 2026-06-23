"""Tests for the atlaslabels module."""

from pathlib import Path

from multiqc import config

from neuroimaging.modules.atlaslabels.atlaslabels import MultiqcModule


DSEG_SAMPLE = """\
0 background 0 0 0 0
1 SFG_L_6_1 0 255 0 255
2 SFG_R_6_1 0 0 255 255
# comment line
3	Precentral_L
4,Frontal_Sup_L
5;Frontal_Sup_R
bad line
189 mAmyg_L 255 120 62 0
190 mAmyg_R 255 117 60 0
"""


# Not testing the actual surface generation as this is done within yabplot
def test_parse_metadata_lines_skips_background():
    regions = MultiqcModule._parse_metadata_lines(DSEG_SAMPLE)
    assert 0 not in regions


def test_parse_metadata_lines_extracts_colors_and_alpha_from_dseg():
    regions = MultiqcModule._parse_metadata_lines(DSEG_SAMPLE)
    assert regions[1]["name"] == "SFG_L_6_1"
    assert regions[1]["r"] == 0
    assert regions[1]["g"] == 255
    assert regions[1]["b"] == 0
    assert regions[1]["a"] == 255
    assert regions[2]["name"] == "SFG_R_6_1"
    assert regions[2]["b"] == 255
    # subcortical: alpha=0 should be captured but NOT filtered yet
    assert regions[189]["a"] == 0
    assert regions[190]["a"] == 0


def test_parse_metadata_lines_supports_common_delimiters():
    regions = MultiqcModule._parse_metadata_lines(DSEG_SAMPLE)
    assert regions[3]["name"] == "Precentral_L"
    assert regions[4]["name"] == "Frontal_Sup_L"
    assert regions[5]["name"] == "Frontal_Sup_R"
    assert len(regions) == 7  # 1,2,3,4,5,189,190 — background and bad line excluded


def test_parse_index_spec_supports_int_ranges_and_mixed_inputs():
    parsed = MultiqcModule._parse_index_spec([1, "2", "5:7", "10-11", "20,21", {"30:31": None}])
    assert parsed == {1, 2, 5, 6, 7, 10, 11, 20, 21, 30, 31}


def test_parse_index_spec_handles_none_and_empty_inputs():
    assert MultiqcModule._parse_index_spec(None) == set()
    assert MultiqcModule._parse_index_spec("") == set()
    assert MultiqcModule._parse_index_spec([]) == set()


def test_build_regions_from_ids_uses_lut_when_present():
    all_regions = {
        1: {"name": "RegionA", "r": 1, "g": 2, "b": 3},
        2: {"name": "RegionB", "r": 4, "g": 5, "b": 6},
    }
    built = MultiqcModule._build_regions_from_ids({1}, all_regions, name_prefix="Cortical")
    assert built == {1: {"name": "RegionA", "r": 1, "g": 2, "b": 3}}


def test_build_regions_from_ids_falls_back_to_synthetic_name_without_lut():
    built = MultiqcModule._build_regions_from_ids({101}, {}, name_prefix="Subcortical")
    assert built == {101: {"name": "Subcortical_101"}}


def test_build_discrete_mapping_prefers_lut_colors_when_complete():
    regions = {
        1: {"name": "A", "r": 255, "g": 0, "b": 0},
        2: {"name": "B", "r": 0, "g": 255, "b": 0},
    }
    values, cmap, vminmax = MultiqcModule._build_discrete_mapping(regions, fallback_cmap_name="plasma")
    assert values == {"A": 1.0, "B": 2.0}
    assert vminmax == [1, 2]
    assert hasattr(cmap, "colors")


def test_build_discrete_mapping_falls_back_to_cmap_when_lut_missing_rgb():
    regions = {
        1: {"name": "A", "r": 255, "g": 0, "b": 0},
        2: {"name": "B"},
    }
    values, cmap, vminmax = MultiqcModule._build_discrete_mapping(regions, fallback_cmap_name="viridis")
    assert values == {"A": 1.0, "B": 2.0}
    assert vminmax == [1, 2]
    assert callable(cmap)


def test_build_discrete_mapping_force_cmap_even_when_rgb_present():
    regions = {
        1: {"name": "A", "r": 255, "g": 0, "b": 0},
        2: {"name": "B", "r": 0, "g": 255, "b": 0},
    }
    values, cmap, vminmax = MultiqcModule._build_discrete_mapping(
        regions,
        fallback_cmap_name="magma",
        force_cmap=True,
    )
    assert values == {"A": 1.0, "B": 2.0}
    assert vminmax == [1, 2]
    assert callable(cmap)


def test_parse_metadata_lines_supports_name_with_spaces():
    text = "1 Left Precentral Region 255 0 0 255"
    regions = MultiqcModule._parse_metadata_lines(text)
    assert regions[1]["name"] == "Left Precentral Region"
    assert regions[1]["r"] == 255
    assert regions[1]["g"] == 0
    assert regions[1]["b"] == 0


def test_parse_metadata_lines_supports_name_only_rows():
    text = "10 Left_Region"
    regions = MultiqcModule._parse_metadata_lines(text)
    assert regions == {10: {"name": "Left_Region"}}


def test_resolve_found_file_path_handles_relative_and_absolute_paths(tmp_path):
    rel = MultiqcModule._resolve_found_file_path({"root": "/tmp/data", "fn": "atlas.nii.gz"})
    assert rel == "/tmp/data/atlas.nii.gz"

    abs_path = Path(tmp_path) / "atlas.nii.gz"
    resolved_abs = MultiqcModule._resolve_found_file_path({"fn": str(abs_path)})
    assert resolved_abs == str(abs_path)


def test_adds_fallback_content_when_atlas_generation_fails(monkeypatch, tmp_path):
    config.kwargs["single_subject"] = True
    config.atlaslabels = {
        "subcortical_rois_indexes": [3, 4],
    }

    def fake_find_log_files(self, pattern):
        if pattern == "atlaslabels/nii":
            return [{"fn": "atlas.nii.gz", "root": str(tmp_path)}]
        if pattern == "atlaslabels/lut":
            return []
        return []

    captured_sections: list[tuple[str, str]] = []

    def fake_add_section(self, name, anchor, content, **kwargs):
        captured_sections.append((name, content))

    def always_fail_build(*args, **kwargs):
        raise RuntimeError("forced atlas build failure")

    monkeypatch.setattr(MultiqcModule, "find_log_files", fake_find_log_files, raising=False)
    monkeypatch.setattr(MultiqcModule, "add_section", fake_add_section, raising=False)
    monkeypatch.setattr(MultiqcModule, "write_data_file", lambda *args, **kwargs: None, raising=False)

    # Patch the module-level yabplot function used in __init__.
    import neuroimaging.modules.atlaslabels.atlaslabels as atlaslabels_mod

    monkeypatch.setattr(atlaslabels_mod.yab, "build_subcortical_atlas", always_fail_build)

    MultiqcModule()

    assert len(captured_sections) == 2
    content_by_name = {name: content for name, content in captured_sections}
    assert content_by_name["Cortical parcellation"] == "<p>Failed to generate cortical atlas preview.</p>"
    assert content_by_name["Subcortical parcellation"] == "<p>Failed to generate subcortical atlas preview.</p>"
