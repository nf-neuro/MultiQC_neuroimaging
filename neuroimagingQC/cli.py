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
