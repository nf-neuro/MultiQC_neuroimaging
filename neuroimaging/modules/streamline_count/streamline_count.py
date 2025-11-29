"""
===============================================
Streamline Count QC Module
===============================================

This module processes streamline count data to provide QC metrics
for neuroimaging pipelines. It displays:
- Streamline count values in general statistics
- Pass/warn/fail status based on IQR outlier detection
"""

import logging
import re
from typing import Dict

import numpy as np

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import violin

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """MultiQC module for streamline count quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Streamline Count",
            anchor="streamline_count",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Visualization of the number of streamlines in a filtered/unfiltered tractogram"
            " using IQR-based outlier detection for quality control.",
        )

        # Check if single-subject mode is enabled
        single_subject_mode = config.kwargs.get("single_subject", False)

        # Halt execution if single-subject mode is enabled
        if single_subject_mode:
            raise ModuleNoSamplesFound

        # Find and parse streamline count files
        sc_data = {}

        config_fp = config.sp.get("streamline_count", {}).get("fn", "")
        for f in self.find_log_files("streamline_count"):
            parsed = self.parse_sc_file(f, config_fp)
            if parsed:
                sample_name = parsed["sample_name"]
                sc_data[sample_name] = parsed["sc_value"]

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names
        sc_data = self.ignore_samples(sc_data)

        if len(sc_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(sc_data)} samples")

        # Add streamline count statistics and plots
        self._add_streamline_count_stats(sc_data)

    def parse_sc_file(self, f, config_fp) -> Dict:
        """
        Parse a streamline count file.

        Expected format:
        8337903
        """
        lines = (f.get("f") or "").splitlines()

        if len(lines) < 1:
            return {}

        # Extract and clean sample name from filename
        # Using the pattern use in custom_code.py for consistency
        # Remove the pattern suffix from filename to get the sample name.
        filename = f["fn"]
        pattern_suffix = config_fp.lstrip("*")
        if pattern_suffix and filename.endswith(pattern_suffix):
            # Remove the suffix and any trailing underscores.
            sample_name = re.sub(r"_+$", "", filename[: -len(pattern_suffix)])
        else:
            # Fallback to default cleaned name if pattern doesn't match
            sample_name = f["s_name"]

        # Apply MultiQC's standard sample name cleaning
        sample_name = self.clean_s_name(sample_name, f)

        # Parse streamline count value
        try:
            sc_value = int(lines[0].strip())
        except (ValueError, TypeError, IndexError):
            return {}

        return {"sample_name": sample_name, "sc_value": sc_value}

    def _add_streamline_count_stats(self, sc_data: Dict[str, int]) -> None:
        """
        Add streamline count statistics and plots.
        """
        # User can configure IQR multiplier via config
        config_thresh = getattr(config, "streamline_count", {})
        iqr_multiplier = config_thresh.get("iqr_multiplier", 3)

        # Calculate IQR-based outliers
        values = np.array(list(sc_data.values()))
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Assign statuses based on IQR outlier detection
        statuses = {}
        for sample_name, sc_value in sc_data.items():
            if sc_value < lower_bound or sc_value > upper_bound:
                statuses[sample_name] = "fail"
            else:
                statuses[sample_name] = "pass"

        # Add streamline count to general statistics
        general_stats_data = {s: {"streamline_count": val} for s, val in sc_data.items()}

        self.general_stats_addcols(
            general_stats_data,
            {
                "streamline_count": {
                    "title": "SC",
                    "description": "Streamline count",
                    "scale": "Blues",
                    "format": "{:,.0f}",
                }
            },
        )

        # Organize statuses into the format expected by add_section
        status_groups = {"pass": [], "fail": []}
        for sample_name, status in statuses.items():
            status_groups[status].append(sample_name)

        # Prepare violin plot data
        # Format: {sample_name: {"Streamline Count": value}}
        plot_data = {}
        for sample_name, sc_value in sc_data.items():
            plot_data[sample_name] = {"Streamline Count": sc_value}

        # Headers for the violin plot
        headers = {
            "Streamline Count": {
                "title": "Streamline Count",
                "description": "Total number of streamlines",
                "scale": "Blues",
                "format": "{:,.0f}",
            }
        }

        # Add inline CSS for proportional status bar widths
        description_html = f"""<style>
.mqc-status-progress-wrapper {{
    width: 100% !important;
    max-width: 100% !important;
}}
.progress-stacked.mqc-status-progress {{
    width: 100% !important;
    max-width: 100% !important;
}}
.progress-stacked.mqc-status-progress .progress {{
    width: 100% !important;
    max-width: 100% !important;
}}
</style>
Tractogram streamline count quality control with outliers detected using the IQR method.
Using an acceptable range defined as IQR * {iqr_multiplier}, subjects with streamline counts
falling outside this range will be flagged, and might indicate potential issues with tractography,
tissue segmentation, or fODF reconstruction. Often times, extremely high streamline counts can be
attributed to a lot of small streamlines generated from noisy fODF peaks. Extremely low streamline counts
may indicate poor white matter segmentation or insufficient seeding. While this might not be sufficient to
exclude a subject, users should investigate such outliers further to ensure data quality.
Pass: within Q1 - {iqr_multiplier}*IQR to Q3 + {iqr_multiplier}*IQR range
[{lower_bound:.0f} - {upper_bound:.0f}], Fail: outside range"""

        self.add_section(
            name="Streamline Count Quality",
            anchor="streamline_count_quality",
            description=description_html,
            plot=violin.plot(
                plot_data,
                headers,
                {
                    "id": "streamline_count_violin",
                    "title": "Streamline Count",
                    "ylab": "Streamline Count",
                },
            ),
            statuses=status_groups,
        )
