#!/usr/bin/env python
"""MultiQC example plugin functions

We can add any custom Python functions here and call them
using the setuptools plugin hooks.
"""

from __future__ import print_function

from multiqc import config
import importlib_metadata
import logging

# Initialise the main MultiQC logger
log = logging.getLogger("multiqc")


# Add default config options for the things that are used in MultiQC_NGI
def neuroimaging_execution_start():
    """Code to execute after the config files and
    command line flags have been parsed.

    This setuptools hook is the earliest that will be able
    to use custom command line flags.
    """

    # Plugin's version number defined in pyproject.toml. Use the package
    # name for this repository and fall back to 'unknown' if metadata
    # isn't available (e.g. when running from a source checkout).
    try:
        version = importlib_metadata.version("neuroimaging")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"
    log.info("Running MultiQC_neuroimaging v{}".format(version))

    # Add to the main MultiQC config object.
    # User config files have already been loaded at this point
    #   so we check whether the value is already set. This is to avoid
    #   clobbering values that have been customised by users.

    # Tractometry search pattern: bundles mean stats TSV
    # log.info("config.sp contents before adding defaults: {}".format(config.sp))
    if "tractometry" not in config.sp:
        config.update_dict(config.sp, {"tractometry": {"fn": "*bundles_mean_stats.tsv"}})

    # Cortical regions search pattern: cortical volume TSV files
    if "cortical/volume" not in config.sp:
        config.update_dict(config.sp, {"cortical/volume": {"fn": "cortical_*_volume_*.tsv"}})

    # Subcortical regions search pattern: subcortical volume TSV files
    if "subcortical/volume" not in config.sp:
        config.update_dict(config.sp, {"subcortical/volume": {"fn": "*_subcortical_volumes.tsv"}})

    # Framewise displacement search pattern: eddy restricted movement RMS files
    if "framewise_displacement" not in config.sp:
        config.update_dict(
            config.sp,
            {"framewise_displacement": {"fn": "*dwi_eddy_restricted_movement_rms.txt"}},
        )

    # Coverage search pattern: dice coefficient files
    if "coverage" not in config.sp:
        config.update_dict(config.sp, {"coverage": {"fn": "*dice.txt"}})

    # Streamline count search pattern
    if "streamline_count" not in config.sp:
        config.update_dict(config.sp, {"streamline_count": {"fn": "*__sc.txt"}})

    # Metricsinroi search pattern
    if "metricsinroi" not in config.sp:
        config.update_dict(config.sp, {"metricsinroi": {"fn": "rois_mean_stats.tsv"}})
