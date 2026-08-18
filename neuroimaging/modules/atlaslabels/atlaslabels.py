"""
=============================
MultiQC Atlas Labels Module
=============================

Project a volumetric atlas to cortical surfaces and visualize it with yabplot.
"""

import base64
from contextlib import redirect_stdout
import io
import logging
import os
import re
import tempfile
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pyvista as pv
import yabplot as yab

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """Project volumetric atlas labels into cortical/subcortical mesh previews."""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Atlas Labels",
            anchor="atlaslabels",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=(
                "This section contains QC images for the segmentation of cortical and subcortical structures, "
                "displayed as an overlay of anatomical labels. These labels are derived from structural MRI and "
                "serve as key regions of interest for connectivity analyses and volumetric measurements. To "
                "assess segmentation accuracy, verify that each label correctly corresponds to its respective "
                "anatomical structure. Too large or too small regions might indicate issues with the segmentation. "
                "A good first step is to assess the alignment of the atlas with the medial wall. If the atlas "
                "is misaligned, it may indicate issues with the atlas mapping process. "
                "If discrepancies are found, consider investigating the segmentation logs. "
                "It is worth noting that the 3D renderings of subcortical structures are generated from "
                "the volumetric atlas and may not perfectly capture the surface details of these structures. "
            ),
        )

        if not config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Fetch our input files and optional LUT
        self.config = getattr(config, "atlaslabels", {})
        self.sp = getattr(config, "sp", {})
        surfaces_files = list(self.find_log_files("atlaslabels/surfaces"))
        annot_files = list(self.find_log_files("atlaslabels/annot"))
        nii_files = list(self.find_log_files("atlaslabels/nii"))
        lut_files = list(self.find_log_files("atlaslabels/lut"))
        if not surfaces_files and not nii_files and not annot_files:
            raise ModuleNoSamplesFound

        # Get the indexes of the cortical and subcortical ROIs from user-defined config.
        # CLI arguments take precedence over config file settings.
        cortical_idx_spec = None
        subcortical_idx_spec = None

        # Check CLI arguments first (take precedence)
        atlas_name_cli = config.kwargs.get("atlas_name")

        cortical_rois_cli = config.kwargs.get("cortical_rois")
        if cortical_rois_cli:
            cortical_idx_spec = list(cortical_rois_cli)

        subcortical_rois_cli = config.kwargs.get("subcortical_rois")
        if subcortical_rois_cli:
            subcortical_idx_spec = list(subcortical_rois_cli)

        # Fallback to config file if CLI not provided
        if atlas_name_cli is None:
            atlas_name_cli = self.config.get("atlas_name")

        if subcortical_idx_spec is None:
            try:
                subcortical_idx_spec = self.config.get("subcortical_rois_indexes")
            except subcortical_idx_spec is None:
                raise ModuleNoSamplesFound("Subcortical ROI indexes must be specified via CLI or config file.")

        # Parse the index specification.
        # If only one of cortical/subcortical specs are there, we assume the remaining
        # IDs are for the other group.
        subcortical_ids = self._parse_index_spec(subcortical_idx_spec)
        if not subcortical_ids:
            log.warning("Subcortical ROI index specs are empty after parsing. Skipping Atlas Labels module.")
            raise ModuleNoSamplesFound

        # Use the LUT files if available
        lut_file = None
        if not lut_files:
            log.warning("No atlas metadata file found. A default colormap will be used.")
            all_regions = {}
        else:
            lut_file = lut_files[0]
            all_regions = self._parse_metadata_lines(lut_file.get("f", ""))

        nii_file = nii_files[0]
        nii_path = self._resolve_found_file_path(nii_file)

        # Assert that we have both lh/rh surfaces and annot files for the specified atlas name.
        # Get the search pattern for the surface files from the config
        sp_surf = self.sp.get("atlaslabels/surfaces", {}).get("fn", "").replace("*.", "")

        # Keep only surfaces that matches lh.{sp_surf} and rh.{sp_surf}
        for f in surfaces_files:
            fn = f.get("fn", "")
            if not re.search(rf"lh\.{sp_surf}$", fn) and not re.search(rf"rh\.{sp_surf}$", fn):
                log.debug(f"Skipping surface file '{fn}' as it does not match the expected pattern.")
            elif re.search(rf"lh\.{sp_surf}$", fn):
                lh_surf_file = self._resolve_found_file_path(f)
            elif re.search(rf"rh\.{sp_surf}$", fn):
                rh_surf_file = self._resolve_found_file_path(f)

        # Extract the two annot files that fit lh.{atlas_name}.annot and rh.{atlas_name}.annot
        lh_annot_file = None
        rh_annot_file = None
        for f in annot_files:
            fn = f.get("fn", "")
            if re.search(rf"lh\.{atlas_name_cli}\.annot$", fn):
                lh_annot_file = self._resolve_found_file_path(f)
            elif re.search(rf"rh\.{atlas_name_cli}\.annot$", fn):
                rh_annot_file = self._resolve_found_file_path(f)

        # Extract the ROIs based on the indexes and fill in the colormap from the LUT if available
        subcortical_regions = self._build_regions_from_ids(subcortical_ids, all_regions, name_prefix="Subcortical")
        cortical_regions = {}

        # Small check to make sure we have valid regions.
        if len(subcortical_regions) == 0:
            log.warning("No valid regions remain after parsing/filtering. Skipping Atlas Labels module.")
            raise ModuleNoSamplesFound

        # Setting temp directories.
        work_dir = Path(tempfile.mkdtemp(prefix="multiqc_atlaslabels_"))
        cortical_dir = Path(self.config.get("cortical_out_dir", str(work_dir / "cortical_atlas")))
        subcortical_dir = Path(self.config.get("subcortical_out_dir", str(work_dir / "subcortical_atlas")))
        cortical_preview_path = work_dir / "cortical_preview.png"
        sub_preview_path = work_dir / "subcortical_preview.png"

        try:
            # Load everything
            lh_vertices, lh_faces = nib.freesurfer.read_geometry(lh_surf_file)
            rh_vertices, rh_faces = nib.freesurfer.read_geometry(rh_surf_file)
            lh_labels, _lh_cmap, _lh_names = nib.freesurfer.read_annot(lh_annot_file)
            rh_labels, _rh_cmap, _rh_names = nib.freesurfer.read_annot(rh_annot_file)

            # get unique labels, but drop NaN values
            unique_labels = np.unique(np.concatenate([lh_labels, rh_labels]))
            unique_labels = unique_labels[~np.isnan(unique_labels)]

            # Build cortical regions dict from the labels
            cortical_regions = self._build_regions_from_ids(unique_labels, all_regions, name_prefix="Cortical")
            # Drop the background (id 0) if present
            cortical_regions.pop(0, None)

            # Convert FreeSurfer mesh data to PyVista PolyData objects
            lh_mesh = self._fs_to_pyvista(lh_vertices, lh_faces)
            rh_mesh = self._fs_to_pyvista(rh_vertices, rh_faces)

            # Small assertion to make sure the number of vertices matches the number of labels
            if len(lh_labels) != lh_mesh.n_points:
                log.warning(
                    "Mismatch in number of vertices and labels for left hemisphere: %d vs %d",
                    len(lh_labels),
                    lh_mesh.n_points,
                )
                raise ModuleNoSamplesFound("Mismatch in number of vertices and labels for left hemisphere.")
            if len(rh_labels) != rh_mesh.n_points:
                log.warning(
                    "Mismatch in number of vertices and labels for right hemisphere: %d vs %d",
                    len(rh_labels),
                    rh_mesh.n_points,
                )
                raise ModuleNoSamplesFound("Mismatch in number of vertices and labels for right hemisphere.")

            # Add the labels as scalars to the meshes
            lh_mesh["Data"] = lh_labels.astype(int)
            rh_mesh["Data"] = rh_labels.astype(int)

            # Build cmap
            _cort_data, cort_cmap, cort_vminmax, cort_colors = self._build_discrete_mapping(
                cortical_regions,
                fallback_cmap_name=self.config.get("cortical_cmap", self.config.get("cmap", "viridis")),
                force_cmap=self.config.get("force_cortical_cmap", False),
            )
            cort_colors.insert(0, (0.99, 0.99, 0.99))  # Very light gray for NaN (background) regions

            # Build the cortical atlas using yabplot
            plotter_cortical = yab.plot_vertexwise(
                lh=lh_mesh,
                rh=rh_mesh,
                scalars="Data",
                lut=cort_colors if cort_colors else cort_cmap,
                vminmax=cort_vminmax,
                nan_color=(0.99, 0.99, 0.99, 1),  # Very light gray for NaN (background) regions
                views=self.config.get(
                    "views",
                    ["left_lateral", "left_medial", "right_medial", "right_lateral"],
                ),
                style=self.config.get("style", "glossy"),
                display_type="matplotlib",
                export_path=str(cortical_preview_path),
            )
            if hasattr(plotter_cortical, "close"):
                plotter_cortical.close()

            # Embed the image as base64 in the report
            if cortical_preview_path.exists():
                cortical_img_b64 = base64.b64encode(cortical_preview_path.read_bytes()).decode("ascii")
                cortical_content = (
                    '<img alt="Cortical atlas mesh preview" '
                    'style="max-width:100%;height:auto;" '
                    f'src="data:image/png;base64,{cortical_img_b64}" />'
                )
        except Exception as e:
            log.warning(f"Cortical atlas build/render failed: {e}")
            cortical_content = "<p>Failed to generate cortical atlas preview.</p>"

        try:
            sub_labels = {rid: info["name"] for rid, info in sorted(subcortical_regions.items())}
            # Redirect log output from yabplot to avoid cluttering the MultiQC report logs with non-critical messages.
            f = io.StringIO()
            with redirect_stdout(f):
                yab.build_subcortical_atlas(
                    nii_path=nii_path,
                    labels_dict=sub_labels,
                    out_dir=str(subcortical_dir),
                    smooth_i=self.config.get("smooth_i", 20),
                    smooth_f=self.config.get("smooth_f", 0.7),
                )
            log.debug(f"yabplot subcortical atlas build output:\n{f.getvalue()}")

            # Similar to the cortical region, build the colormap either based on the LUT
            # or the default colormap.
            sub_plot_regions = dict(subcortical_regions)
            sub_data, sub_cmap, sub_vminmax, sub_colors = self._build_discrete_mapping(
                sub_plot_regions,
                fallback_cmap_name=self.config.get("subcortical_cmap", self.config.get("cmap", "viridis")),
                force_cmap=self.config.get("force_subcortical_cmap", False),
            )

            f = io.StringIO()
            with redirect_stdout(f):
                plotter_sub = yab.plot_subcortical(
                    data=sub_data,
                    custom_atlas_path=str(subcortical_dir),
                    bmesh=None,
                    views=self.config.get(
                        "views",
                        ["left_lateral", "left_medial", "superior", "anterior"],
                    ),
                    cmap=sub_colors if sub_colors else sub_cmap,
                    vminmax=sub_vminmax,
                    style=self.config.get("style", "glossy"),
                    display_type="matplotlib",
                    export_path=str(sub_preview_path),
                )
            log.debug(f"yabplot subcortical atlas plot output:\n{f.getvalue()}")

            if hasattr(plotter_sub, "close"):
                plotter_sub.close()

            # Embed the image as base64 in the report
            if sub_preview_path.exists():
                sub_img_b64 = base64.b64encode(sub_preview_path.read_bytes()).decode("ascii")
                subcortical_content = (
                    '<img alt="Subcortical atlas mesh preview" '
                    'style="max-width:100%;height:auto;" '
                    f'src="data:image/png;base64,{sub_img_b64}" />'
                )
        except Exception as e:
            log.warning(f"Subcortical atlas build/render failed: {e}")
            subcortical_content = "<p>Failed to generate subcortical atlas preview.</p>"

        # Add the sections ot the report using base MultiQC functions
        self.add_section(
            name="Cortical parcellation",
            anchor="atlaslabels-cortical-preview",
            content=cortical_content,
        )

        self.add_section(
            name="Subcortical parcellation",
            anchor="atlaslabels-subcortical-preview",
            content=subcortical_content,
        )

        self.write_data_file(
            {
                "atlas_nifti": nii_path,
                "atlas_metadata_source": lut_file.get("fn", "") if lut_file else "",
                "cortical_output_dir": str(cortical_dir),
                "cortical_region_count": len(cortical_regions),
                "cortical_index_spec": cortical_idx_spec,
                "subcortical_region_count": len(subcortical_regions),
                "subcortical_output_dir": str(subcortical_dir),
                "subcortical_index_spec": subcortical_idx_spec,
            },
            "multiqc_atlaslabels",
        )

    @staticmethod
    def _fs_to_pyvista(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
        """
        Convert FreeSurfer mesh data to a PyVista PolyData object.
        """
        vtk_faces = np.hstack(
            [
                np.full((faces.shape[0], 1), 3, dtype=np.int32),
                faces.astype(np.int32),
            ]
        )
        return pv.PolyData(vertices, vtk_faces)

    @staticmethod
    def _resolve_found_file_path(found_file: dict) -> str:
        fn = found_file.get("fn", "")
        if os.path.isabs(fn):
            return fn
        root = found_file.get("root", "")
        return os.path.join(root, fn) if root else fn

    @staticmethod
    def _build_discrete_mapping(
        regions: dict[int, dict], fallback_cmap_name: str = "viridis", force_cmap: bool = False
    ) -> tuple[dict[str, float], ListedColormap | object, list[float], str]:
        """Build deterministic region->value map and colors.

        Uses LUT RGB values only when every region has RGB fields.
        Otherwise, falls back to a matplotlib colormap.
        """
        values: dict[str, float] = {}
        colors = []
        has_complete_rgb = bool(regions) and all(all(k in info for k in ("r", "g", "b")) for info in regions.values())

        for idx, (rid, info) in enumerate(sorted(regions.items()), start=1):
            values[info["name"]] = float(idx)
            if has_complete_rgb and not force_cmap:
                r, g, b = int(info["r"]), int(info["g"]), int(info["b"])
                colors.append((r / 255.0, g / 255.0, b / 255.0))

        if not colors:
            cmap = colormaps.get_cmap(fallback_cmap_name)
            vminmax = [1, max(1, len(values))]
            return values, cmap, vminmax, False

        cmap = ListedColormap(colors)
        vminmax = [1, len(colors)]
        return values, cmap, vminmax, colors

    @staticmethod
    def _parse_index_spec(index_spec) -> set[int]:
        """Parse flexible ROI index specification from config.

        Supported examples:
          - 12
          - "12"
          - "1:188" (inclusive)
          - "1-188" (inclusive)
          - [1, 2, "5:10", "20-30"]
          - [{"1:188": null}] (defensive support for odd YAML flow parsing)
        """
        parsed: set[int] = set()

        def _parse_one(item):
            if item is None:
                return
            if isinstance(item, int):
                parsed.add(item)
                return
            if isinstance(item, str):
                token = item.strip()
                if not token:
                    return
                range_match = re.match(r"^(-?\d+)\s*[:-]\s*(-?\d+)$", token)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    step = 1 if end >= start else -1
                    parsed.update(range(start, end + step, step))
                    return
                if re.match(r"^-?\d+$", token):
                    parsed.add(int(token))
                    return
                for part in token.split(","):
                    if part.strip() != token:
                        _parse_one(part)
                return
            if isinstance(item, dict):
                for k, v in item.items():
                    _parse_one(k)
                    _parse_one(v)
                return
            if isinstance(item, (list, tuple, set)):
                for sub in item:
                    _parse_one(sub)

        _parse_one(index_spec)
        return parsed

    @staticmethod
    def _build_regions_from_ids(
        selected_ids: set[int], all_regions: dict[int, dict], name_prefix: str
    ) -> dict[int, dict]:
        """Build a region dict from selected IDs, falling back to synthetic names when LUT is missing."""
        regions: dict[int, dict] = {}
        for rid in sorted(selected_ids):
            if rid in all_regions:
                regions[rid] = dict(all_regions[rid])
            else:
                regions[rid] = {"name": f"{name_prefix}_{rid}"}
        return regions

    @staticmethod
    def _parse_metadata_lines(text: str) -> dict[int, dict]:
        """Parse atlas metadata lines into {id: {name, r, g, b}}.

        Supported formats (delimiter: whitespace, comma, or semicolon):
          - BIDS dseg / ITK-SNAP: ``id  name  R  G  B  A``
          - RGB only:             ``id  name  R  G  B``
          - Name only:            ``id  name``
          - Names with spaces:    ``id  Left Region  R  G  B  A``

        Background (id 0) is always skipped.
        """
        regions: dict[int, dict] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [t for t in re.split(r"[\s,;]+", line) if t]
            if len(tokens) < 2:
                continue
            try:
                region_id = int(tokens[0])
            except ValueError:
                continue
            if region_id == 0:
                continue  # skip background
            # Try to detect a trailing RGBA (4 tokens) or RGB (3 tokens) block.
            color: dict = {}
            name_end = len(tokens)
            for n_color in (4, 3):
                if len(tokens) >= 2 + n_color:
                    try:
                        vals = [int(t) for t in tokens[-n_color:]]
                        color = {"r": vals[0], "g": vals[1], "b": vals[2]}
                        if n_color == 4:
                            color["a"] = vals[3]
                        name_end = len(tokens) - n_color
                        break
                    except ValueError:
                        pass
            region_name = " ".join(tokens[1:name_end]).strip("\",'")
            if region_name:
                regions[region_id] = {"name": region_name, **color}
        return regions
