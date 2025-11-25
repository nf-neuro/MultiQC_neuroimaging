"""
===============================================
Coverage QC Module
===============================================

This module processes coverage/dice coefficient data to provide QC metrics
for neuroimaging pipelines. It displays:
- Dice coefficient values in general statistics
- Pass/warn/fail status based on thresholds
"""

import logging
import re
from typing import Dict

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound
from multiqc.plots import violin

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """MultiQC module for coverage quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Coverage",
            anchor="coverage",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Assessment of the white matter coverage of tractograms using the DICE coefficient "
            "between the tract density map and white matter mask for quality control.",
        )

        # Check if single-subject mode is enabled
        single_subject_mode = config.kwargs.get("single_subject", False)

        # Halt execution if single-subject mode is enabled
        if single_subject_mode:
            raise ModuleNoSamplesFound

        # Find and parse dice files
        dice_data = {}

        for f in self.find_log_files("coverage"):
            parsed = self.parse_dice_file(f)
            if parsed:
                sample_name = parsed["sample_name"]
                dice_data[sample_name] = parsed["dice_value"]

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by sample names
        dice_data = self.ignore_samples(dice_data)

        if len(dice_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(dice_data)} samples")

        # Add coverage statistics and plots
        self._add_coverage_stats(dice_data)

    def parse_dice_file(self, f) -> Dict:
        """
        Parse a dice coefficient file.

        Expected format:
        0.8593300982298845
        """
        lines = (f.get("f") or "").splitlines()

        if len(lines) < 1:
            return {}

        # Extract and clean sample name from filename
        # The filename is like: sub-1019__dice.txt
        # We want to extract just the subject ID: sub-1019
        filename = f["fn"]
        match = re.search(r"(sub-[A-Za-z0-9]+)", filename)
        if match:
            sample_name = match.group(1)
        else:
            # Fallback to default cleaned name if pattern doesn't match
            sample_name = f["s_name"]

        # Apply MultiQC's standard sample name cleaning
        sample_name = self.clean_s_name(sample_name, f)

        # Parse dice coefficient value
        try:
            dice_value = float(lines[0].strip())
        except (ValueError, TypeError, IndexError):
            return {}

        return {"sample_name": sample_name, "dice_value": dice_value}

    def _add_coverage_stats(self, dice_data: Dict[str, float]) -> None:
        """
        Add coverage statistics and plots.
        """
        # User can configure thresholds via config
        config_thresh = getattr(config, "coverage", {})
        warn_threshold = config_thresh.get("warn_threshold", 0.9)
        fail_threshold = config_thresh.get("fail_threshold", 0.8)

        # Assign statuses based on dice thresholds
        statuses = {}
        for sample_name, dice_value in dice_data.items():
            if dice_value < fail_threshold:
                statuses[sample_name] = "fail"
            elif dice_value < warn_threshold:
                statuses[sample_name] = "warn"
            else:
                statuses[sample_name] = "pass"

        # Add dice coefficient to general statistics
        general_stats_data = {s: {"dice_coefficient": val} for s, val in dice_data.items()}

        self.general_stats_addcols(
            general_stats_data,
            {
                "dice_coefficient": {
                    "title": "Dice Coef.",
                    "description": "Dice coefficient for coverage",
                    "max": 1.0,
                    "min": 0.0,
                    "scale": "RdYlGn",
                    "format": "{:,.3f}",
                }
            },
        )

        # Organize statuses into the format expected by add_section
        status_groups = {"pass": [], "warn": [], "fail": []}
        for sample_name, status in statuses.items():
            status_groups[status].append(sample_name)

        # Prepare violin plot data
        # Format: {sample_name: {"Dice": value}}
        plot_data = {}
        for sample_name, dice_value in dice_data.items():
            plot_data[sample_name] = {"Dice": dice_value}

        # Headers for the violin plot
        headers = {
            "Dice": {
                "title": "Dice Coefficient",
                "description": "Dice coefficient for coverage",
                "max": 1.0,
                "min": 0.0,
                "scale": "RdYlGn",
                "format": "{:,.3f}",
            }
        }

        # Add inline CSS for proportional status bar widths
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
White matter coverage by the tractogram measured by computing the DICE coefficient
between the subject's tract density map and its corresponding white matter mask.
Higher DICE coefficients indicate better coverage of the white matter. Users should expect
values over 0.9 for unfiltered tractograms and over 0.8 for filtered tractograms.
Pass: >0.9, Warn: 0.8-0.9, Fail: <0.8"""

        self.add_section(
            name="Coverage Quality",
            anchor="coverage_quality",
            description=description_html,
            plot=violin.plot(
                plot_data,
                headers,
                {
                    "id": "coverage_dice_violin",
                    "title": "Coverage: Dice Coefficients",
                    "ylab": "Dice Coefficient",
                },
            ),
            statuses=status_groups,
        )
