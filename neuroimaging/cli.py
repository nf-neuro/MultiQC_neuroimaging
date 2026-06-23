#!/usr/bin/env python
"""
MultiQC command line options - we tie into the MultiQC
core here and add some new command line parameters.

See the Click documentation for more command line flag types:
http://click.pocoo.org/5/
"""

import click

# Sets config.kwargs['single_subject'] to True if specified (will be False
# otherwise)
single_subject = click.option(
    "--single-subject-report",
    "single_subject",
    is_flag=True,
    help="Generate a single subject report (disables multi-subject modules).",
)

# Set of customizable options for the atlaslabels module, that might be
# useful for subject-level dynamic config.
subcortical_rois = click.option(
    "--subcortical-rois",
    "subcortical_rois",
    type=click.STRING,
    multiple=True,
    help="Subcortical ROI indexes or intervals to include in the atlaslabels module. "
    "Can specify multiple times (e.g., '--subcortical-rois 1-100 --subcortical-rois 150-200') "
    "or use comma-separated values in a single argument (e.g., '--subcortical-rois 1-100,150-200'). "
    "Ranges are inclusive (e.g., '1-100' includes ROIs 1 and 100).",
)
atlas_name = click.option(
    "--atlas-name",
    "atlas_name",
    type=click.STRING,
    help="Name of the atlas to use in the atlaslabels module. This is used for labeling and display purposes.",
)
