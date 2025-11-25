"""
===============================================
Cortical Regions QC Module
===============================================

This module processes cortical region volume data to provide QC metrics
for neuroimaging pipelines. It displays:
- Percentage of successfully extracted regions in general statistics
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
    """MultiQC module for cortical region extraction quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Cortical Regions",
            anchor="cortical",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Assessment of cortical region volumes for quality control using IQR-based outlier detection."
            " Each cortical region's volume is evaluated across subjects, and regions with volumes "
            "falling outside the range defined by Q1 - 3*IQR to Q3 + 3*IQR are considered outliers."
            " The percentage of outlier regions per subject is reported in the general statistics, "
            "with thresholds for pass/warn/fail configurable in the MultiQC configuration file.",
        )

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Get configuration
        self.cortical_config = getattr(config, "cortical", {})
        warn_threshold = self.cortical_config.get("warn_threshold", 20)
        fail_threshold = self.cortical_config.get("fail_threshold", 10)

        # Find and parse cortical volume files
        cortical_data = {}

        for f in self.find_log_files("cortical/volume"):
            parsed = self.parse_cortical_file(f)
            if parsed:
                # Merge regions from multiple files (lh and rh) for same
                # samples
                for sample_name, regions_dict in parsed.items():
                    if sample_name not in cortical_data:
                        cortical_data[sample_name] = {}
                    cortical_data[sample_name].update(regions_dict)

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names
        cortical_data = self.ignore_samples(cortical_data)

        if len(cortical_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(cortical_data)} samples")

        # Calculate outlier percentages for each sample
        sample_percentages = self._calculate_outlier_percentages(cortical_data)

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
                    "description": "Percentage of cortical regions with volumes outside 3*IQR range",
                    "suffix": "%",
                    "max": max_outlier_pct,
                    "min": 0,
                    "scale": "RdYlGn-rev",
                    "format": "{:,.1f}",
                }
            },
        )

        # Add violin plots for volume distributions
        self._add_per_region_plots(cortical_data, status_data)

        # Write parsed data to file
        self.write_data_file(cortical_data, "multiqc_cortical_data")

    def parse_cortical_file(self, f) -> Dict:
        """
        Parse a cortical volume TSV file.

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

    def _calculate_outlier_percentages(self, cortical_data: Dict) -> Dict[str, float]:
        """
        Calculate the percentage of outlier regions per sample.

        For each region, computes IQR across all subjects and identifies
        outliers as values outside Q1 - 3*IQR to Q3 + 3*IQR range.

        Args:
            cortical_data: Dict mapping sample names to region volumes

        Returns:
            Dict mapping sample names to outlier percentages
        """
        # Organize data by region
        region_values = {}
        for sample_name, regions_dict in cortical_data.items():
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

        for sample_name, regions_dict in cortical_data.items():
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
        cortical_data: Dict,
        status_data: Dict,
    ) -> None:
        """
        Add violin plots showing volume distribution per region.
        Each plot shows all regions with their volume distributions.
        """
        # Group regions by hemisphere
        lh_regions = {}
        rh_regions = {}

        # First, reorganize data to group by hemisphere
        for sample_name, regions_dict in cortical_data.items():
            for region_name, volume in regions_dict.items():
                if region_name.startswith("lh_"):
                    if region_name not in lh_regions:
                        lh_regions[region_name] = {}
                    lh_regions[region_name][sample_name] = volume
                elif region_name.startswith("rh_"):
                    if region_name not in rh_regions:
                        rh_regions[region_name] = {}
                    rh_regions[region_name][sample_name] = volume

        # Convert to format needed for violin plots
        # Format: {sample: {region: volume}}
        lh_plot_data = {}
        rh_plot_data = {}

        for sample_name in cortical_data.keys():
            lh_plot_data[sample_name] = {}
            rh_plot_data[sample_name] = {}

            for region_name in lh_regions.keys():
                if region_name in cortical_data[sample_name]:
                    lh_plot_data[sample_name][region_name] = cortical_data[sample_name][region_name]

            for region_name in rh_regions.keys():
                if region_name in cortical_data[sample_name]:
                    rh_plot_data[sample_name][region_name] = cortical_data[sample_name][region_name]

        # Configuration for each hemisphere section
        regions_config = [
            {
                "plot_data": lh_plot_data,
                "regions": lh_regions,
                "name": "Left Hemisphere Volume Distribution",
                "anchor": "cortical_lh_volumes",
                "id": "cortical_lh_volume_plot",
                "title": "Cortical Regions: Left Hemisphere Volume Distribution",
                "description": "Distribution of cortical region volumes in "
                "the left hemisphere across all samples. You may look for extreme outliers, which "
                "could indicate segmentation issues or data quality problems. Combined with other indicators, "
                "these outliers may help identify subjects that require further investigation or exclusion.",
            },
            {
                "plot_data": rh_plot_data,
                "regions": rh_regions,
                "name": "Right Hemisphere Volume Distribution",
                "anchor": "cortical_rh_volumes",
                "id": "cortical_rh_volume_plot",
                "title": "Cortical Regions: Right Hemisphere Volume Distribution",
                "description": "Distribution of cortical region volumes in "
                "the right hemisphere across all samples. You may look for extreme outliers, which "
                "could indicate segmentation issues or data quality problems. Combined with other indicators, "
                "these outliers may help identify subjects that require further investigation or exclusion.",
            },
        ]

        for region_cfg in regions_config:
            plot_data = region_cfg["plot_data"]

            if not plot_data or not region_cfg["regions"]:
                continue

            # Create headers for regions
            headers = {
                region: {
                    "title": region,
                    "description": f"Volume for {region}",
                }
                for region in region_cfg["regions"].keys()
            }

            # Add inline CSS for full-width status bars
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
{region_cfg["description"]}"""

            self.add_section(
                name=region_cfg["name"],
                anchor=region_cfg["anchor"],
                description=description_html,
                plot=violin.plot(
                    plot_data,
                    headers,
                    {
                        "id": region_cfg["id"],
                        "title": region_cfg["title"],
                        "ylab": "Volume (mm³)",
                        "xlab": "Cortical Regions",
                    },
                ),
                statuses=status_data,
            )
