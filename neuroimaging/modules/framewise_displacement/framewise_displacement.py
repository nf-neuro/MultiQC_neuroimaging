"""
===============================================
Framewise Displacement QC Module
===============================================

This module processes framewise displacement data to provide QC metrics
for neuroimaging pipelines. It displays:
- Maximum FD value in general statistics (multi-subject mode)
- Line plot with colored regions based on FD thresholds
- Single subject line plot (single-subject mode)
"""

import logging
import re
from typing import Dict

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import linegraph

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """MultiQC module for framewise displacement quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Framewise Displacement",
            anchor="framewise_displacement",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Assessment of subject motion during acquisition using the framewise displacement (FD) metric "
            "calculated by FSL's `eddy` tool for quality control.",
        )

        # Check if single-subject mode is enabled
        single_subject_mode = config.kwargs.get("single_subject", False)

        # Find and parse FD files
        fd_data = {}

        config_fp = config.sp.get("framewise_displacement", {}).get("fn", "")
        for f in self.find_log_files("framewise_displacement"):
            parsed = self.parse_fd_file(f, config_fp)
            if parsed:
                sample_name = parsed["sample_name"]
                fd_data[sample_name] = parsed["values"]

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names
        fd_data = self.ignore_samples(fd_data)

        if len(fd_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(fd_data)} samples")

        if single_subject_mode:
            # Single-subject mode: plot line plot for single subject
            self._add_single_subject_plot(fd_data)
        else:
            # Multi-subject mode: calculate max FD and create plots
            self._add_multi_subject_plots(fd_data)

    def parse_fd_file(self, f, config_fp) -> Dict:
        """
        Parse a framewise displacement file.

        Expected format:
        col1  col2
        0.1   0.15
        0.2   0.25
        ...
        """
        lines = (f.get("f") or "").splitlines()

        if len(lines) < 1:
            return {}

        # Extract and clean sample name from filename
        # Using the pattern use in custom_code.py for consistency
        # Remove the pattern suffix from filename to get the sample name.
        filename = f["fn"]
        pattern_suffix = config_fp.lstrip("*")  # Remove leading *
        if pattern_suffix and filename.endswith(pattern_suffix):
            # Remove the suffix to get the sample name part (without remaining "_" if any, can have
            # multiple underscores)
            sample_name = re.sub(r"_+$", "", filename[: -len(pattern_suffix)])  # Remove trailing underscores
        else:
            # Fallback to default cleaned name if pattern doesn't match
            sample_name = f["s_name"]

        # Apply MultiQC's standard sample name cleaning
        sample_name = self.clean_s_name(sample_name, f)

        # Parse second column values
        values = []
        for line in lines:
            if not line.strip():
                continue

            fields = line.strip().split()
            if len(fields) < 2:
                continue

            try:
                value = float(fields[1])
                values.append(value)
            except (ValueError, TypeError):
                continue

        if not values:
            return {}

        return {"sample_name": sample_name, "values": values}

    def _add_single_subject_plot(self, fd_data: Dict) -> None:
        """
        Add line plot for single subject.
        """
        if len(fd_data) != 1:
            log.warning(f"Expected 1 sample in single-subject mode, found {len(fd_data)}")

        # Get the single sample data
        sample_name = list(fd_data.keys())[0]
        values = fd_data[sample_name]

        # Calculate max for y-axis
        max_value = max(values) if values else 1.0
        y_max = max(max_value * 1.1, 0.8)  # At least 0.8 for visibility

        # Create plot data: {sample: {x: y}}
        plot_data = {sample_name: {i: value for i, value in enumerate(values)}}

        # Fetch from config thresholds
        config_thresh = getattr(config, "framewise_displacement", {})
        warn_threshold = config_thresh.get("warn_threshold", 0.8)
        fail_threshold = config_thresh.get("fail_threshold", 2.0)

        # Add colored regions
        plot_config = {
            "id": "fd_single_subject_plot",
            "title": "Framewise Displacement: Single Subject",
            "ylab": "FD (mm)",
            "xlab": "Volume",
            "y_bands": [
                {"from": 0, "to": warn_threshold, "color": "#C5FFC5"},
                {"from": warn_threshold, "to": fail_threshold, "color": "#FFFF99"},
                {"from": fail_threshold, "to": 10, "color": "#FFB6C6"},
            ],
            "y_lines": [
                {
                    "value": warn_threshold,
                    "color": "#95a5a6",
                    "dash": "dash",
                    "label": f"Threshold: {warn_threshold} mm",
                },
                {
                    "value": fail_threshold,
                    "color": "#95a5a6",
                    "dash": "dash",
                    "label": f"Threshold: {fail_threshold} mm",
                },
            ],
            "colors": {sample_name: "#000000"},
            "ymax": y_max,
        }

        self.add_section(
            name="Framewise Displacement",
            anchor="fd_single_subject",
            description="Framewise displacement across volumes. "
            "Each point represents the FD for a volume compared to the previous volume. "
            "High spikes in FD may indicate excessive motion during scanning. "
            "While `eddy` attempts to correct for motion, users should be cautious "
            "when interpreting data from subjects with high FD values. "
            f"Green: <{warn_threshold}mm, Yellow: {warn_threshold}-{fail_threshold}mm, Red: >{fail_threshold}mm",
            plot=linegraph.plot(plot_data, plot_config),
        )

    def _add_multi_subject_plots(self, fd_data: Dict) -> None:
        """
        Add plots and statistics for multi-subject analysis.
        """
        # User can configure thresholds via config
        config_thresh = getattr(config, "framewise_displacement", {})
        warn_threshold = config_thresh.get("warn_threshold", 0.8)
        fail_threshold = config_thresh.get("fail_threshold", 2.0)

        # Calculate max FD for each sample
        max_fd_values = {}
        all_values = []
        for sample_name, values in fd_data.items():
            if values:
                max_fd_values[sample_name] = max(values)
                all_values.extend(values)

        # Assign colors and statuses based on max FD thresholds
        colors = {}
        statuses = {}
        for sample_name, max_fd in max_fd_values.items():
            if max_fd < warn_threshold:
                colors[sample_name] = "#2ecc71"  # Green
                statuses[sample_name] = "pass"
            elif max_fd < fail_threshold:
                colors[sample_name] = "#f39c12"  # Yellow/Orange
                statuses[sample_name] = "warn"
            else:
                colors[sample_name] = "#e74c3c"  # Red
                statuses[sample_name] = "fail"

        # Add max FD to general statistics
        general_stats_data = {s: {"max_fd": max_val} for s, max_val in max_fd_values.items()}

        # Get max value for scale
        max_fd_overall = max(max_fd_values.values()) if max_fd_values else 2.0

        self.general_stats_addcols(
            general_stats_data,
            {
                "max_fd": {
                    "title": "Max FD",
                    "description": "Maximum framewise displacement",
                    "suffix": " mm",
                    "max": max_fd_overall,
                    "min": 0,
                    "scale": "RdYlGn-rev",
                    "format": "{:,.2f}",
                }
            },
        )

        # Calculate y-axis max for plot
        y_max = max(all_values) * 1.1 if all_values else 1.0
        y_max = max(y_max, 0.8)  # At least 0.8 for visibility

        # Create line plot with all subjects
        plot_data = {}
        for sample_name, values in fd_data.items():
            plot_data[sample_name] = {i: value for i, value in enumerate(values)}

        # Plot config without bands, with colored lines
        plot_config = {
            "id": "fd_multi_subject_plot",
            "title": "Framewise Displacement",
            "ylab": "FD (mm)",
            "xlab": "Volume",
            "colors": colors,
            "y_lines": [
                {
                    "value": warn_threshold,
                    "color": "#000000",
                    "dash": "dash",
                    "label": f"Threshold: {warn_threshold} mm",
                },
                {
                    "value": fail_threshold,
                    "color": "#000000",
                    "dash": "dash",
                    "label": f"Threshold: {fail_threshold} mm",
                },
            ],
            "ymax": y_max,
        }

        # Organize statuses into the format expected by add_section
        status_groups = {"pass": [], "warn": [], "fail": []}
        for sample_name, status in statuses.items():
            status_groups[status].append(sample_name)

        self.add_section(
            name="Framewise Displacement",
            anchor="fd_multi_subject",
            description="Framewise displacement (FD) across volumes for all "
            "subjects. Each line represents a subject and its relative "
            "movement over time. For each volume, the FD"
            "represents the amount of movement from the previous volume. "
            "Subjects with big spikes in FD may indicate excessive motion "
            "during scanning. While `eddy` attempts to correct for motion, "
            "users should be cautious when interpreting data from subjects "
            "with high FD values. "
            "Lines are colored based on maximum FD. "
            f"Green: <{warn_threshold}mm (pass), "
            f"Yellow: {warn_threshold}-{fail_threshold}mm (warn), "
            f"Red: >{fail_threshold}mm (fail)",
            plot=linegraph.plot(plot_data, plot_config),
            statuses=status_groups,
        )
