"""
===============================================
Subcortical Regions QC Module
===============================================

This module processes subcortical region volume data to provide QC metrics
for neuroimaging pipelines. It displays:
- Percentage of outlier regions in general statistics
- Distribution of volumes across regions with violin plots
"""

import logging
from typing import Dict

import numpy as np

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import violin

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """MultiQC module for subcortical region extraction quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Subcortical Regions",
            anchor="subcortical",
            href="https://github.com/nf-neuro/MultiQC-neuroimaging",
            info="Quality control for subcortical region segmentation",
        )

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Get configuration
        self.subcortical_config = getattr(config, "subcortical", {})
        warn_threshold = self.subcortical_config.get("warn_threshold", 20)
        fail_threshold = self.subcortical_config.get("fail_threshold", 10)

        # Find and parse subcortical volume files
        subcortical_data = {}

        for f in self.find_log_files("subcortical/volume"):
            parsed = self.parse_subcortical_file(f)
            if parsed:
                for sample_name, regions_dict in parsed.items():
                    if sample_name not in subcortical_data:
                        subcortical_data[sample_name] = {}
                    subcortical_data[sample_name].update(regions_dict)

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names
        subcortical_data = self.ignore_samples(subcortical_data)

        if len(subcortical_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(subcortical_data)} samples")

        # Calculate outlier percentages for each sample
        sample_percentages = self._calculate_outlier_percentages(subcortical_data)

        # Create status bar data
        # Note: Lower outlier percentage is better
        status_data = {"pass": [], "warn": [], "fail": []}
        for sample_name, percentage in sample_percentages.items():
            if percentage <= fail_threshold:
                status_data["pass"].append(sample_name)
            elif percentage <= warn_threshold:
                status_data["warn"].append(sample_name)
            else:
                status_data["fail"].append(sample_name)

        # Add region percentage to general statistics
        general_stats_data = {s: {"region_pct": pct} for s, pct in sample_percentages.items()}

        # Get max value for scale
        max_outlier_pct = max(sample_percentages.values()) if sample_percentages else 100

        self.general_stats_addcols(
            general_stats_data,
            {
                "region_pct": {
                    "title": "% Outliers",
                    "description": "Percentage of subcortical regions with volumes outside 3*IQR range",
                    "suffix": "%",
                    "max": max_outlier_pct,
                    "min": 0,
                    "scale": "RdYlGn-rev",
                    "format": "{:,.1f}",
                }
            },
        )

        # Add violin plots for volume distributions
        self._add_per_region_plots(subcortical_data, status_data)

        # Write parsed data to file
        self.write_data_file(subcortical_data, "multiqc_subcortical_data")

    def parse_subcortical_file(self, f) -> Dict:
        """
        Parse a subcortical volume TSV file.

        Expected format:
        Sample  region1  region2  region3  ...
        sub-001 1234.5   2345.6   3456.7   ...
        sub-002 1111.1   2222.2   3333.3   ...
        """
        data = {}
        lines = (f.get("f") or "").splitlines()

        if len(lines) < 2:
            return data

        # Parse header
        headers = lines[0].strip().split("\t")
        region_names = headers[1:]  # Skip "Sample" column

        # Parse data rows
        for line in lines[1:]:
            if not line.strip():
                continue

            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue

            sample_name = fields[0]
            volumes = fields[1:]

            # Create dict with region: volume pairs
            sample_data = {}
            for region_name, volume_str in zip(region_names, volumes):
                try:
                    volume = float(volume_str)
                    sample_data[region_name] = volume
                except (ValueError, TypeError):
                    sample_data[region_name] = 0.0

            data[sample_name] = sample_data

        return data

    def _calculate_outlier_percentages(self, subcortical_data: Dict) -> Dict[str, float]:
        """
        Calculate the percentage of outlier regions per sample.

        For each region, computes IQR across all subjects and identifies
        outliers as values outside Q1 - 3*IQR to Q3 + 3*IQR range.

        Args:
            subcortical_data: Dict mapping sample names to region volumes

        Returns:
            Dict mapping sample names to outlier percentages
        """
        # Organize data by region
        region_values = {}
        for sample_name, regions_dict in subcortical_data.items():
            for region_name, volume in regions_dict.items():
                if region_name not in region_values:
                    region_values[region_name] = []
                region_values[region_name].append(volume)

        # Calculate IQR bounds for each region
        region_iqr_bounds = {}
        for region_name, values in region_values.items():
            values_array = np.array(values)

            if len(values_array) < 4:
                # Use full range if too few samples
                region_iqr_bounds[region_name] = (
                    float(np.min(values_array)),
                    float(np.max(values_array)),
                )
                continue

            # Calculate Q1, Q3, and IQR using numpy
            q1, q3 = np.percentile(values_array, [25, 75])
            iqr = q3 - q1

            # Define outlier bounds: Q1 - 3*IQR and Q3 + 3*IQR
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            region_iqr_bounds[region_name] = (lower_bound, upper_bound)

        # Count outlier regions per sample
        sample_percentages = {}
        total_regions = len(region_iqr_bounds)

        for sample_name, regions_dict in subcortical_data.items():
            outlier_count = 0
            for region_name, volume in regions_dict.items():
                if region_name in region_iqr_bounds:
                    lower_bound, upper_bound = region_iqr_bounds[region_name]
                    # Count if volume is outside the IQR bounds
                    if volume < lower_bound or volume > upper_bound:
                        outlier_count += 1

            # Calculate percentage of outlier regions
            percentage = (outlier_count / total_regions * 100) if total_regions > 0 else 0.0
            sample_percentages[sample_name] = percentage

        return sample_percentages

    def _add_per_region_plots(
        self,
        subcortical_data: Dict,
        status_data: Dict,
    ) -> None:
        """
        Add violin plot showing volume distribution per region.
        """
        # Convert to format needed for violin plots
        # Format: {sample: {region: volume}}
        plot_data = subcortical_data

        # Create headers for regions
        if plot_data:
            first_sample = list(plot_data.keys())[0]
            headers = {
                region: {
                    "title": region,
                    "description": f"Volume for {region}",
                }
                for region in plot_data[first_sample].keys()
            }

            # Add inline CSS for full-width status bars
            description_html = """<style>
.mqc-status-progress-wrapper {
    width: 100% !important;
    max-width: 100% !important;
}
.progress-stacked.mqc-status-progress {
    width: 100% !important;
    max-width: 100% !important;
}
.progress-stacked.mqc-status-progress .progress {
    width: 100% !important;
    max-width: 100% !important;
}
</style>
Distribution of subcortical region volumes across all samples."""

            self.add_section(
                name="Subcortical Volume Distribution",
                anchor="subcortical_volumes",
                description=description_html,
                plot=violin.plot(
                    plot_data,
                    headers,
                    {
                        "id": "subcortical_volume_plot",
                        "title": "Subcortical Regions: Volume Distribution",
                        "ylab": "Volume (mm³)",
                        "xlab": "Subcortical Regions",
                    },
                ),
                statuses=status_data,
            )
