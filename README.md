![Static Badge](https://img.shields.io/badge/python-%3E3.8-blue?logo=python)
[![Code Lint](https://github.com/nf-neuro/MultiQC_neuroimaging/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/nf-neuro/MultiQC_neuroimaging/actions/workflows/lint.yml)
[![Plugin tests](https://github.com/nf-neuro/MultiQC_neuroimaging/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/nf-neuro/MultiQC_neuroimaging/actions/workflows/test.yml)

# MultiQC_neuroimaging

A MultiQC plugin for comprehensive quality control of neuroimaging pipelines. This plugin aggregates QC metrics from various neuroimaging analysis outputs into interactive HTML reports.

## Features

The plugin includes modules for:

- **Cortical Regions**: QC metrics for cortical volume segmentation with IQR-based outlier detection
- **Subcortical Regions**: QC metrics for subcortical volume segmentation with IQR-based outlier detection
- **Tractometry**: Bundle extraction quality assessment with FA, volume, and streamline count metrics
- **Coverage**: Dice coefficient validation for tractography WM coverage
- **Framewise Displacement**: Head motion assessment between DWI volumes
- **Streamline Count**: Tractography quality control using IQR-based outlier detection

All modules feature configurable thresholds, status indicators (pass/warn/fail), and interactive visualizations.

## Installation

```bash
pip install multiqc
pip install git+https://github.com/nf-neuro/MultiQC_neuroimaging.git
```

For development:

```bash
pip install multiqc
git clone https://github.com/nf-neuro/MultiQC_neuroimaging.git
cd neuroimaging
pip install -e ".[dev]"
pre-commit install
```

> [!NOTE]
>
> `pre-commit install` will setup hooks to ensure your code respects the coding standards. Those hooks will
> be run each time you commit a file. We aligned our standards with those of `MultiQC`, for more information
> see [their documentation](https://docs.seqera.io/multiqc/development/modules#code-formatting).

## Usage

Run MultiQC on your neuroimaging pipeline outputs:

```bash
multiqc /path/to/analysis/directory
```

The plugin will automatically detect and parse compatible files based on predefined search patterns.

### Configuration

Customize thresholds in your MultiQC config file and select the order of modules:

```yaml
framewise_displacement:
  warn_threshold: 0.8 # mm
  fail_threshold: 2.0 # mm

coverage:
  warn_threshold: 0.9
  fail_threshold: 0.8

streamline_count:
  iqr_multiplier: 3

tractometry:
  warn_threshold: 90
  fail_threshold: 80

cortical:
  warn_threshold: 20
  fail_threshold: 10

subcortical:
  warn_threshold: 20
  fail_threshold: 10

module_order:
  - framewise_displacement
  - coverage
  - streamline_count
  - tractometry
  - cortical
  - subcortical
```

## Contribute a new module

Contribution are welcomed! Creating a module in this plugin follows the same instructions as creating a module directly in MultiQC. For detailed instructions, visit the [MultiQC documentation](https://docs.seqera.io/multiqc/development/modules).

## Documentation

For more information on MultiQC plugins, visit the [MultiQC documentation](https://docs.seqera.io/multiqc/development/plugins).
