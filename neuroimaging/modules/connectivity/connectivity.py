"""
===========================================
Connectivity Module
===========================================

This module collects and visualizes connectivity matrices
fromm connectome analysis. It supports the following file types:
    * NumPy (.npy) connectivity matrices (for now)

For global reports, it computes the density of the connectivity
matrice, and plots the distribution across subjects in a violin plot.
It also generates a heatmap of the average connectivity matrix
across all subjects.

For subject reports, it visualizes the individual connectivity
matrix as a heatmap.
"""

import json
import logging
import re
from typing import Dict

import numpy as np
import pandas as pd

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import heatmap

log = logging.getLogger(__name__)


VIRIDIS_COLSTOPS = [
    [0.0, "#440154"],
    [0.25, "#3b528b"],
    [0.5, "#21918c"],
    [0.75, "#5ec962"],
    [1.0, "#fde725"],
]


class MultiqcModule(BaseMultiqcModule):
    """ "MultiQC module for connectivity matrices."""

    def __init__(self):
        #  Get the single-subject mode if set.
        single_subject_mode = config.kwargs.get("single_subject", False)

        # Tailor the module description based on the mode
        if single_subject_mode:
            module_info = (
                "Visualization of individual structural connectivity matrices for detailed quality inspection. "
                "Each connectivity matrix is displayed as a heatmap showing diffusion metric between brain regions. "
                "Assess the matrix for bilateral symmetry, anatomical plausibility, isolated regions, and unexpected artifacts. "
                "The density (proportion of non-zero connections) provides a summary metric for overall connectivity. "
                "If multiple diffusion metrics (FA, MD, RD, AD) are available, separate heatmaps are generated for comparison."
            )
        else:
            module_info = (
                "Comprehensive cohort-level assessment of structural connectivity matrices for quality control. "
                "For each subject, the density of the connectivity matrix (proportion of non-zero connections) is computed "
                "to identify subjects with unusually sparse or dense connectomes that may indicate processing issues. "
                "Subjects are automatically flagged based on statistical outlier detection (IQR method: fail beyond 1.5×IQR, warn beyond Q1/Q3). "
                "The frequency matrix visualizes connection consistency across all subjects, helping identify "
                "systematic connectivity patterns and potential anatomical or processing artifacts. "
                "High-frequency connections indicate consistent structural pathways, while extremely low frequencies may suggest tractography or thresholding issues."
            )

        super(MultiqcModule, self).__init__(
            name="Structural Connectivity",
            anchor="connectivity",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info=module_info,
        )

        # Find and parse connectivity matrix files.
        # If no metrics are configured, include every metric detected in input filenames.
        module_config = getattr(config, "connectivity", {}) or {}
        metrics_filter = self._parse_metrics_filter(module_config.get("metrics", None))
        conn_data = {}
        for f in self.find_log_files("connectivity/matrices"):
            metric = self._extract_metric_from_filename(f["fn"])
            if metric is None:
                log.debug(f"Could not infer connectivity metric from filename '{f['fn']}'. Skipping file.")
                continue
            if metrics_filter is not None and metric not in metrics_filter:
                continue

            parsed = self.parse_connectivity_file(f)
            if parsed:
                sample_name = parsed["sample_name"]
                # Support multiple metrics per sample
                if sample_name not in conn_data:
                    conn_data[sample_name] = {}
                conn_data[sample_name][metric] = parsed["values"]

        # Find LUT files.
        for f in self.find_log_files("connectivity/lut"):
            # Look if the file is a JSON file or a TSV file
            if f["fn"].lower().endswith(".json"):
                with open(f["root"] + "/" + f["fn"], "r") as fp:
                    lut = json.load(fp)
            elif f["fn"].lower().endswith(".tsv"):
                lut_df = pd.read_csv(f["root"] + "/" + f["fn"], sep="\t")
                # by BIDS convention, first column is index and second column is label
                lut = dict(zip(lut_df.iloc[:, 0].astype(str), lut_df.iloc[:, 1].astype(str)))
            else:
                log.debug(f"Unsupported LUT file format for '{f['fn']}'. Expected .json or .tsv. Skipping file.")
                raise ModuleNoSamplesFound(f"Unsupported LUT file format for '{f['fn']}'. Expected .json or .tsv.")

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names.
        conn_data = self.ignore_samples(conn_data)

        if len(conn_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(conn_data)} samples")

        # Generate global plots if not in single-subject mode
        if not single_subject_mode:
            # Compute densities - use the first available metric for each sample
            densities = {
                s_name: {"density": self.compute_density(next(iter(metrics.values())))}
                for s_name, metrics in conn_data.items()
            }

            # Identify outliers based on density (e.g. using X * IQR rule)
            config_thresh = getattr(config, "connectivity", {})
            iqr_multiplier = config_thresh.get("iqr_multiplier", 1.5)
            density_values = [d["density"] for d in densities.values()]
            q1, q3 = np.percentile(density_values, [25, 75])
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            for s_name, d in densities.items():
                if d["density"] < lower_bound or d["density"] > upper_bound:
                    d["flag"] = "fail"
                elif d["density"] < q1 or d["density"] > q3:
                    d["flag"] = "warn"
                else:
                    d["flag"] = "pass"

            status_groups = {"pass": [], "warn": [], "fail": []}
            for s_name, d in densities.items():
                status_groups[d["flag"]].append(s_name)

            # Add to general stats table.
            self.general_stats_addcols(
                densities,
                {
                    "density": {
                        "title": "Density",
                        "description": "Density of the connectivity matrix (proportion of non-zero connections).",
                        "min": min(d["density"] for d in densities.values()) - 0.1,
                        "max": max(d["density"] for d in densities.values()) + 0.1,
                        "format": "{:.2f}",
                    }
                },
            )
            # Generate the frequency matrix and plot it as a heatmap.
            freq_mat = self.frequency_matrix(conn_data, tuple((len(lut), len(lut))))
            self.add_section(
                name="Connectivity Frequency Matrix",
                description="This heatmap displays the frequency of connections across all subjects in the cohort. "
                "Each cell (i, j) represents the proportion of subjects that have a non-zero connection between regions i and j. "
                "Values range from 0 (purple, connection never present) to 1 (yellow, connection present in all subjects). ",
                plot=heatmap.plot(
                    freq_mat,
                    xcats=[lut.get(str(i + 1)) for i in range(freq_mat.shape[0])],
                    pconfig={
                        "id": "connectivity_frequency_matrix",
                        "title": "Connectivity Frequency Matrix",
                        "display_values": False,
                        "xcats_samples": False,
                        "ycats_samples": False,
                        # Mimic a viridis colormap.
                        "colstops": VIRIDIS_COLSTOPS,
                    },
                ),
                statuses=status_groups,
            )
        else:
            # In single-subject mode, render one matrix section controlled by a metric selector.
            # We select the first sample because single-subject mode should only contain one sample.
            sample_name, metrics_dict = next(iter(conn_data.items()))
            metrics = sorted(metrics_dict)
            default_metric = metrics[0]

            if len(metrics) > 1:
                self.add_section(
                    name="Connectivity Metric Selection",
                    anchor="connectivity_metric_selection",
                    description="Select the diffusion metric used to display the connectivity matrix.",
                    content=self._build_metric_selector(metrics, default_metric, sample_name),
                )

            self.add_section(
                name="Connectivity Matrix",
                anchor="connectivity_matrix",
                description=f"Individual structural connectivity matrix for {sample_name}. "
                "Each cell (i, j) represents the diffusion metric between brain regions i and j. "
                "Values are displayed using a viridis colormap where purple indicates low/no connectivity and yellow indicates high connectivity.",
                content=self._build_metric_matrices(metrics_dict, lut, sample_name, default_metric),
            )

    def _heatmap_pconfig(self, plot_id: str, title: str) -> Dict[str, object]:
        """Return common heatmap configuration for connectivity plots."""
        return {
            "id": plot_id,
            "title": title,
            "display_values": False,
            "xcats_samples": False,
            "ycats_samples": False,
            "colstops": VIRIDIS_COLSTOPS,
        }

    def _parse_metrics_filter(self, metrics_config):
        """Normalize metrics configuration.

        Returns None when all detected metrics should be included.
        """
        if metrics_config in (None, [], "all"):
            return None
        if isinstance(metrics_config, str):
            return {metrics_config.lower()}
        return {metric.lower() for metric in metrics_config}

    def _extract_metric_from_filename(self, filename: str):
        """Extract metric token from connectivity matrix filenames."""
        match = re.search(r"stat-(.+?)\.npy$", filename, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1).lower()

    def _build_metric_selector(self, metrics: list, default_metric: str, sample_name: str) -> str:
        """Build dropdown HTML for selecting which metric heatmap is visible."""
        options_html = "".join(
            [
                f'<option value="{metric}" {"selected" if metric == default_metric else ""}>{metric.upper()}</option>'
                for metric in metrics
            ]
        )
        hook_name = f"renderConnectivityMetric_{re.sub(r'[^0-9a-zA-Z_]', '_', sample_name)}"
        return f"""
        <div class="d-flex justify-content-center">
            <div class="text-center">
                <label class="form-label mb-2 fw-semibold">Metric</label>
                <select
                    class="form-select shadow-sm"
                    aria-label="Connectivity metric selection"
                    onchange="if (typeof {hook_name} === 'function') {hook_name}(this.value)"
                    style="min-width: 180px;"
                >
                    {options_html}
                </select>
            </div>
        </div>
        """

    def _build_metric_matrices(
        self,
        metrics_dict: Dict[str, np.ndarray],
        lut: Dict[str, str],
        sample_name: str,
        default_metric: str,
    ) -> str:
        """Build a single section containing all metric heatmaps and JS to toggle them."""
        sample_id = re.sub(r"[^0-9a-zA-Z_]", "_", sample_name)
        metric_plot_ids = {metric: f"{sample_name}_{metric}_connectivity_matrix" for metric in sorted(metrics_dict)}
        content = ""
        for metric, plot_id in metric_plot_ids.items():
            matrix = metrics_dict[metric]
            display = "block" if metric == default_metric else "none"
            plot = heatmap.plot(
                matrix,
                xcats=[lut.get(str(i + 1)) for i in range(matrix.shape[0])],
                pconfig=self._heatmap_pconfig(plot_id=plot_id, title=f"{metric.upper()} Connectivity Matrix"),
            )
            content += f"""
            <div id="{plot_id}_container" style="display: {display};">
                {plot.interactive_plot(self.anchor, "connectivity_matrix")}
            </div>
            """

        hook_name = f"renderConnectivityMetric_{sample_id}"
        containers_var = f"connectivity_metric_containers_{sample_id}"
        content += f"""
        <script>
        var {containers_var} = {json.dumps(metric_plot_ids)};
        function {hook_name}(metric) {{
            Object.values({containers_var}).forEach(function(plotId) {{
                document.getElementById(plotId + '_container').style.display = 'none';
            }});

            var selectedPlotId = {containers_var}[metric];
            var selectedContainer = document.getElementById(selectedPlotId + '_container');
            selectedContainer.style.display = 'block';

            if (typeof renderPlot === 'function') {{
                renderPlot(selectedPlotId);
            }}

            setTimeout(function() {{
                var heatmapContainer = selectedContainer.querySelector('.mqc-heatmap-container');
                if (heatmapContainer) {{
                    heatmapContainer.dispatchEvent(new Event('resize'));
                }}
            }}, 0);
        }}
        </script>
        """

        return content

    def parse_connectivity_file(self, f: str) -> Dict:
        """Parse a connectivity matrix file.

        Args:
            f (str): Path to the connectivity matrix file.
            config_fp (str): Configured file path pattern to identify connectivity files.

        Returns:
            Dict: Parsed data including sample name and connectivity values.
        """
        values = np.load(f["root"] + "/" + f["fn"])

        # Extract and clean sample name from filename.
        # Since patterns can vary between atlases and stats, we should manually match
        # patterns with sub-XXXXX, ses-XXXX, and run-ZZZ assuming BIDS structure.
        match = re.search(r"sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?(_run-[a-zA-Z0-9]+)?", f["fn"])
        if match:
            sample_name = match.group(0)
        elif "s_name" in f:
            # If the sample name is already provided in the file metadata, use it.
            sample_name = f["s_name"]
        elif "fn" in f:
            # As a last resort, use the filename as the sample name.
            sample_name = f["fn"]

        # Apply MultiQC's sample name cleaning
        sample_name = self.clean_s_name(sample_name, f)

        return {"sample_name": sample_name, "values": values}

    def compute_density(self, matrix: np.ndarray) -> float:
        """Compute the density of a connectivity matrix.

        Args:
            matrix (np.ndarray): Connectivity matrix.

        Returns:
            float: Density of the connectivity matrix (proportion between 0 and 1).
        """
        # Count non-zero connections
        non_zero_connections = np.count_nonzero(matrix)
        total_connections = matrix.size
        # Compute density and rescale to be between 0 and 1
        density = non_zero_connections / total_connections if total_connections > 0 else 0

        return density

    def frequency_matrix(self, matrices: Dict[str, Dict[str, np.ndarray]], shape: tuple = None) -> np.ndarray:
        """Compute the frequency of all connections across subjects.

        Args:
            matrices (Dict[str, Dict[str, np.ndarray]]): Dictionary of sample names and their metrics dictionaries.
            shape (tuple): Expected shape of the connectivity matrices.

        Returns:
            np.ndarray: Frequency matrix where each element represents the proportion of subjects
                        with this connection present.
        """
        # Create empty zero matrix to store results.
        # Get the first matrix from the first sample's first metric
        first_matrix = next(iter(next(iter(matrices.values())).values()))
        freq_mat = np.zeros(shape if shape else first_matrix.shape)

        for sub, metrics_dict in matrices.items():
            # Use the first available metric for each subject
            matrix = next(iter(metrics_dict.values()))
            # Small check to ensure matrices are the expected shape.
            if matrix.shape != freq_mat.shape:
                log.warning(f"Matrix for subject {sub} has shape {matrix.shape}, expected {freq_mat.shape}. Skipping.")
                continue
            freq_mat += np.where(matrix != 0, 1, 0)  # Increment by 1 where connection is present

        return freq_mat / len(matrices)  # Rescale to be between 0 and 1
