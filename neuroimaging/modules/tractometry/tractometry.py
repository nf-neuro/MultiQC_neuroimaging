"""MultiQC module for tractometry results.

This module looks for files named `bundles_mean_stats.tsv` and builds:
- Violin plot showing distribution of percentages of bundles detected per
  sample.
- Violin plot showing FA, volume, and streamline counts per bundle across
  samples.
"""

import csv
import logging
from typing import Dict, Any

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import violin

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")


class MultiqcModule(BaseMultiqcModule):
    """Module to parse `bundles_mean_stats.tsv` and present QC metrics."""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Tractometry",
            anchor="tractometry",
            href="https://github.com/scilus/nf-pediatric",
            info="This section contains QC metrics from tractometry analysis"
            + ". For QC purposes, they only include fractional anisotropy"
            + " (FA), volume, and streamline for the whole bundles. "
            + "Additional metrics can be found in the statistics table "
            + "exported by the pipeline. The status bars indicate flagged subjects "
            + "based on the percentage of bundles detected, with thresholds "
            + "configurable in the MultiQC configuration file.",
        )

        # Halt execution if single-subject mode is enabled
        if config.kwargs.get("single_subject", False):
            raise ModuleNoSamplesFound

        # Find files using the custom search pattern added in custom_code
        files = list(self.find_log_files("tractometry"))

        # Nothing found - raise ModuleNoSamplesFound to tell MultiQC
        if len(files) == 0:
            log.debug(f"Could not find tractometry reports in {config.analysis_dir}")
            raise ModuleNoSamplesFound

        # Parse data from TSV files
        samples_bundles: Dict[str, set] = {}
        bundle_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for f in files:
            content = f.get("f", "")
            sname = f.get("s_name")
            reader = csv.DictReader(content.splitlines(), delimiter="\t")

            for row in reader:
                # Prefer per-row 'sample' column over file-level s_name
                row_sample = (row.get("sample") or "").strip()
                sample = row_sample if row_sample else (sname or "unknown")
                bundle = row.get("bundle") or "unnamed"

                samples_bundles.setdefault(sample, set()).add(bundle)

                # Collect FA, volume, streamlines_count for each bundle/sample
                bundle_metrics.setdefault(bundle, {})
                bundle_metrics[bundle].setdefault(sample, {})

                for metric in ["fa", "volume", "streamlines_count"]:
                    if metric in row and row[metric]:
                        try:
                            val = float(row[metric])
                            bundle_metrics[bundle][sample][metric] = val
                        except (ValueError, TypeError):
                            pass

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter ignored samples
        samples_bundles = self.ignore_samples(samples_bundles)

        if len(samples_bundles) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(samples_bundles)} samples")

        # Compute per-sample bundle counts and percentages
        sample_counts = {s: len(b) for s, b in samples_bundles.items()}
        total_bundles = len(bundle_metrics)
        sample_percentages = {
            s: (count / total_bundles * 100) if total_bundles > 0 else 0 for s, count in sample_counts.items()
        }

        # Create status categories based on bundle percentages
        # User can configure thresholds via config
        config_thresh = getattr(config, "tractometry", {})
        warn_threshold = config_thresh.get("warn_threshold", 90)
        fail_threshold = config_thresh.get("fail_threshold", 80)

        passed = []
        warned = []
        failed = []
        for s, pct in sample_percentages.items():
            if pct < fail_threshold:
                failed.append(s)
            elif pct < warn_threshold:
                warned.append(s)
            else:
                passed.append(s)

        status_data = {
            "pass": passed,
            "warn": warned,
            "fail": failed,
        }

        # Add bundle percentage to general statistics table
        general_stats_data = {s: {"bundle_percentage": pct} for s, pct in sample_percentages.items()}

        self.general_stats_addcols(
            general_stats_data,
            {
                "bundle_percentage": {
                    "title": "% Bundles",
                    "description": "Percentage of bundles detected",
                    "suffix": "%",
                    "max": 100,
                    "min": 0,
                    "scale": "RdYlGn",
                    "format": "{:,.1f}",
                }
            },
        )

        # Create violin plots for FA, volume, streamlines per bundle
        self._add_per_bundle_plots(bundle_metrics, status_data)

        # Write parsed data to file
        self.write_data_file(
            {"sample_counts": sample_counts, "bundles": bundle_metrics},
            "multiqc_tractometry",
        )

    def _add_per_bundle_plots(
        self,
        bundle_metrics: Dict[str, Dict[str, Dict[str, Any]]],
        status_data: Dict[str, list],
    ) -> None:
        """Create violin plots for FA, volume, and streamlines per bundle."""

        # Organize data: for each metric, create {bundle: {metric_key: value}}
        # Violin plot shows distribution of metric values across samples
        # for each bundle
        metrics_config = [
            {
                "key": "fa",
                "title": "Fractional Anisotropy (FA)",
                "xlab": "FA",
                "description": "Distribution of FA values per bundle. "
                + "You should look for extreme outliers, as it "
                + "can be indicative of issues in bundle extraction.",
            },
            {
                "key": "volume",
                "title": "Bundle Volume",
                "xlab": "Volume (mm³)",
                "description": "Distribution of volume per bundle. You "
                + "should look for extreme outliers, as it can "
                + "indicate issues in bundle extraction. "
                + "Bundles with very low volume may be incomplete "
                + "while bundles with high volume might include "
                + "spurious streamlines or non-desirable structures.",
            },
            {
                "key": "streamlines_count",
                "title": "Streamline Count",
                "xlab": "# Streamlines",
                "description": "Distribution of streamline counts per bundle"
                + ". You should look for extreme outliers, as it can "
                + "indicate issues in bundle extraction. Too high "
                + "streamline counts may indicate inclusion of "
                + "spurious streamlines, while too low counts may "
                + "indicate incomplete bundles.",
            },
        ]

        for metric_cfg in metrics_config:
            metric_key = metric_cfg["key"]

            # Restructure data: samples as rows, bundles as columns
            # Format: {sample_name: {bundle_name: metric_value}}
            plot_data = {}
            headers = {}

            # Collect all bundles that have this metric
            for bundle in sorted(bundle_metrics.keys()):
                samples_data = bundle_metrics[bundle]

                for sample, metrics in samples_data.items():
                    if metric_key in metrics:
                        if sample not in plot_data:
                            plot_data[sample] = {}
                        plot_data[sample][bundle] = metrics[metric_key]

                # Create header for this bundle column
                headers[bundle] = {
                    "title": bundle,
                    "description": f"{metric_cfg['title']} for {bundle}",
                }

            # Skip if no data for this metric
            if not plot_data:
                continue

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
{metric_cfg["description"]}"""

            # Create single violin plot with all bundles as columns
            self.add_section(
                name=metric_cfg["title"],
                anchor=f"tractometry-{metric_key}",
                description=description_html,
                plot=violin.plot(
                    plot_data,
                    headers=headers,
                    pconfig={
                        "id": f"tractometry_{metric_key}_violin",
                        "title": f"Tractometry: {metric_cfg['title']}",
                    },
                ),
                statuses=status_data,
            )
