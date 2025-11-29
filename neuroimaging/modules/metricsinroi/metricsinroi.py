"""MultiQC module for metricsinroi results.

This module looks for files named `rois_mean_stats.tsv` and builds:
- Violin plot showing distribution of mean FA per roi across samples.
"""

import csv
import logging
from typing import Dict, Any

import numpy as np

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import violin

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")


class MultiqcModule(BaseMultiqcModule):
    """Module to parse `rois_mean_stats.tsv` and present QC metrics."""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="metricsinroi",
            anchor="metricsinroi",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="This section contains QC metrics from metrics in region-of-interests analysis"
            + ". For QC purposes, it only includes fractional anisotropy"
            + " (FA). "
            + "Additional metrics can be found in the statistics table "
            + "exported by the pipeline. Subjects are flagged using "
            + "IQR-based outlier detection on mean FA values for quality control.",
        )

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Find files using the custom search pattern added in custom_code
        files = list(self.find_log_files("metricsinroi"))

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(files) == 0:
            log.debug(f"Could not find metricsinroi reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        # Parse data from TSV files
        samples_rois: Dict[str, set] = {}
        roi_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for f in files:
            content = f.get("f", "")
            sname = f.get("s_name")
            reader = csv.DictReader(content.splitlines(), delimiter="\t")

            for row in reader:
                # Prefer per-row 'sample' column over file-level s_name
                row_sample = (row.get("sample") or "").strip()
                sample = row_sample if row_sample else (sname or "unknown")
                roi = row.get("roi") or "unnamed"

                samples_rois.setdefault(sample, set()).add(roi)

                # Collect FA, volume, streamlines_count for each roi/sample
                roi_metrics.setdefault(roi, {})
                roi_metrics[roi].setdefault(sample, {})

                for metric in ["fa"]:
                    if metric in row and row[metric]:
                        try:
                            val = float(row[metric])
                            roi_metrics[roi][sample][metric] = val
                        except (ValueError, TypeError):
                            pass

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter ignored samples
        samples_rois = self.ignore_samples(samples_rois)

        if len(samples_rois) == 0:
            raise ModuleNoSamplesFound

        # Also remove ignored samples from roi_metrics
        for roi in roi_metrics:
            for sample in list(roi_metrics[roi].keys()):
                if sample not in samples_rois:
                    del roi_metrics[roi][sample]

        log.info(f"Found {len(samples_rois)} samples")

        # Create status categories based on FA IQR outlier detection per ROI
        # User can configure IQR multiplier via config
        config_thresh = getattr(config, "metricsinroi", {})
        iqr_multiplier = config_thresh.get("iqr_multiplier", 3)

        # Calculate IQR-based outliers for FA per ROI
        # A sample fails if it's an outlier in ANY roi
        passed = set(samples_rois.keys())
        failed = set()

        roi_bounds = {}  # Store bounds for each ROI for reporting

        for roi, samples_data in roi_metrics.items():
            # Get FA values for this ROI across all samples
            roi_fa_values = []
            roi_samples = []
            for sample in samples_rois.keys():
                if sample in samples_data and "fa" in samples_data[sample]:
                    roi_fa_values.append(samples_data[sample]["fa"])
                    roi_samples.append(sample)

            if len(roi_fa_values) > 0:
                values = np.array(roi_fa_values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr

                roi_bounds[roi] = (lower_bound, upper_bound)

                # Check each sample for this ROI
                for sample, fa_value in zip(roi_samples, roi_fa_values):
                    if fa_value < lower_bound or fa_value > upper_bound:
                        failed.add(sample)
                        if sample in passed:
                            passed.remove(sample)

        status_data = {
            "pass": list(passed),
            "warn": [],
            "fail": list(failed),
        }

        # Create violin plots for FA per roi
        self._add_per_roi_plots(roi_metrics, status_data, iqr_multiplier, roi_bounds)

        # Write parsed data to file
        self.write_data_file(
            {"rois": roi_metrics},
            "multiqc_metricsinroi",
        )

    def _add_per_roi_plots(
        self,
        roi_metrics: Dict[str, Dict[str, Dict[str, Any]]],
        status_data: Dict[str, list],
        iqr_multiplier: float,
        roi_bounds: Dict[str, tuple],
    ) -> None:
        """Create violin plots for FA per roi."""

        # Organize data: for each metric, create {roi: {metric_key: value}}
        # Violin plot shows distribution of metric values across samples
        # for each roi
        metrics_config = [
            {
                "key": "fa",
                "title": "Fractional Anisotropy (FA)",
                "xlab": "FA",
                "description": "Distribution of FA values per roi. "
                + "You should look for extreme outliers, as it "
                + "can be indicative of issues in roi extraction.",
            },
        ]

        for metric_cfg in metrics_config:
            metric_key = metric_cfg["key"]

            # Restructure data: samples as rows, rois as columns
            # Format: {sample_name: {roi_name: metric_value}}
            plot_data = {}
            headers = {}

            # Collect all rois that have this metric
            for roi in sorted(roi_metrics.keys()):
                samples_data = roi_metrics[roi]

                for sample, metrics in samples_data.items():
                    if metric_key in metrics:
                        if sample not in plot_data:
                            plot_data[sample] = {}
                        plot_data[sample][roi] = metrics[metric_key]

                # Create header for this roi column
                headers[roi] = {
                    "title": roi,
                    "description": f"{metric_cfg['title']} for {roi}",
                }

            # Skip if no data for this metric
            if not plot_data:
                continue

            # Add inline CSS for full-width status bars
            if metric_key == "fa":
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
{metric_cfg["description"]} Quality control uses IQR-based outlier detection on mean FA values for each ROI independently.
Using an acceptable range defined as IQR * {iqr_multiplier}, subjects with FA values
falling outside this range in ANY ROI will be flagged.
Pass: within Q1 - {iqr_multiplier}*IQR to Q3 + {iqr_multiplier}*IQR range for all ROIs, Fail: outside range in at least one ROI"""

            # Create single violin plot with all rois as columns
            self.add_section(
                name=metric_cfg["title"],
                anchor=f"metricsinroi-{metric_key}",
                description=description_html,
                plot=violin.plot(
                    plot_data,
                    headers=headers,
                    pconfig={
                        "id": f"metricsinroi_{metric_key}_violin",
                        "title": f"MetricsInROI: {metric_cfg['title']}",
                    },
                ),
                statuses=status_data,
            )
