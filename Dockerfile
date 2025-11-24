FROM python:3.13-slim

# Optional pandoc installation for PDF support
ARG INSTALL_PANDOC=false

# - Install `ps` for Nextflow
# - Install MultiQC from PyPI
# - Add custom group and user
RUN \
    echo "Docker build log: Run apt-get update" 1>&2 && \
    apt-get update -y -qq \
    && \
    echo "Docker build log: Install procps" 1>&2 && \
    apt-get install -y -qq procps && \
    if [ "$INSTALL_PANDOC" = "true" ]; then \
        echo "Docker build log: Install pandoc and LaTeX for PDF generation" 1>&2 && \
        apt-get install -y -qq pandoc texlive-latex-base texlive-fonts-recommended texlive-latex-extra texlive-luatex; \
    fi && \
    echo "Docker build log: Clean apt cache" 1>&2 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y && \
    echo "Docker build log: Upgrade pip and install multiqc from PyPI" 1>&2 && \
    pip install --quiet --upgrade pip && \
    pip install multiqc && \
    echo "Docker build log: Add multiqc user and group" 1>&2 && \
    groupadd --gid 1000 multiqc && \
    useradd -ms /bin/bash --create-home --gid multiqc --uid 1000 multiqc

# Copy the neuroimaging plugin source
COPY . /tmp/neuroimaging/

# Install the neuroimaging plugin (regular install, not editable)
RUN echo "Docker build log: Install neuroimaging plugin" 1>&2 && \
    pip install --no-cache-dir /tmp/neuroimaging && \
    echo "Docker build log: Delete python cache directories" 1>&2 && \
    find /usr/local/lib/python3.13 \( -iname '*.c' -o -iname '*.pxd' -o -iname '*.pyd' -o -iname '__pycache__' \) -exec rm -rf {} + 2>/dev/null || true && \
    echo "Docker build log: Delete /tmp/neuroimaging" 1>&2 && \
    rm -rf /tmp/neuroimaging

# Set to be the new user
USER multiqc

# Set default workdir to user home
WORKDIR /home/multiqc

# Check everything is working smoothly
RUN echo "Docker build log: Testing multiqc" 1>&2 && \
    multiqc --help

# Set multiqc as the entrypoint
ENTRYPOINT ["multiqc"]

# Default to showing help if no arguments provided
CMD ["--help"]