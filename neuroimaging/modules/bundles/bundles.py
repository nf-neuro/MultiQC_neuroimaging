"""
===============================================
Bundles QC Module
===============================================

This module provides interactive 3D visualization of WM bundles
using Niivue for quality assessment.
"""

import logging
import os
import re
from typing import Dict

from multiqc import config
from multiqc.base_module import BaseMultiqcModule, ModuleNoSamplesFound

log = logging.getLogger(__name__)


class MultiqcModule(BaseMultiqcModule):
    """MultiQC module for bundle visualization quality control"""

    def __init__(self):
        super(MultiqcModule, self).__init__(
            name="Bundles",
            anchor="bundles",
            href="https://github.com/nf-neuro/MultiQC_neuroimaging",
            info="Visualization of white matter bundles for quality control.",
        )

        # Check if single-subject mode is enabled
        single_subject_mode = config.kwargs.get("single_subject", False) or getattr(
            config, "single_subject_report", False
        )
        log.info(f"Single subject mode: {single_subject_mode}")
        log.info(f"config.kwargs: {config.kwargs}")
        log.info(f"config.single_subject_report: {getattr(config, 'single_subject_report', None)}")

        # Halt execution if single-subject mode is NOT enabled
        if not single_subject_mode:
            raise ModuleNoSamplesFound

        # Find and parse bundle files
        bundle_data = {}

        config_fp = config.sp.get("bundles", {}).get("fn", "")
        log.info(f"Looking for files with pattern: {config_fp}")

        files_found = list(self.find_log_files("bundles", filecontents=False, filehandles=True))
        log.info(f"Found {len(files_found)} files matching pattern")

        for f in files_found:
            log.info(f"Processing file: {f['fn']}")
            parsed = self.parse_bundle_file(f, config_fp)
            if parsed:
                bundle_name = parsed["bundle_name"]
                bundle_data[bundle_name] = parsed["filepath"]
                log.info(f"Added bundle: {bundle_name}")

        # Superfluous function call to confirm that it is used in this module
        # Replace None with actual version if it is available
        self.add_software_version(None)

        # Filter by bundle names.
        bundle_data = self.ignore_samples(bundle_data)

        if len(bundle_data) == 0:
            raise ModuleNoSamplesFound

        log.info(f"Found {len(bundle_data)} bundles")

        # Add bundle visualization section
        self._add_bundle_visualizations(bundle_data)

    def parse_bundle_file(self, f, config_fp) -> Dict:
        """
        Parse a dictionary containing bundle information. This function basically
        is used to extract bundle name and filepath from the file dictionary.

        Parameters
        ----------
        f : dict
            A file dictionary from MultiQC containing file content and metadata.
        config_fp : str
            The expected filename pattern from the MultiQC config.

        Returns
        -------
        Dict
            A dictionary with parsed bundle information.
        """

        # Each bundle file is named as <sub-info>__<bundle_name>_labels_uniformized.trk
        # Example: sub-01_ses-01__PYT_L_labels_uniformized.trk
        filename = f["fn"]
        pattern_suffix = config_fp.lstrip("*")
        if pattern_suffix and filename.endswith(pattern_suffix):
            bundle_part = filename[: -len(pattern_suffix)]
            parts = bundle_part.split("__")
            if len(parts) == 2:
                bundle_name = re.sub(r"_+$", "", parts[1])  # Remove trailing underscores
                filepath = os.path.join(f["root"], f["fn"])
            else:
                bundle_name = parts[0]  # Fallback to first part if no "__" found
                filepath = os.path.join(f["root"], f["fn"])
        else:
            # Fallback to default cleaned name if pattern doesn't match
            bundle_name = f["s_name"]
            filepath = os.path.join(f["root"], f["fn"])

        return {"bundle_name": bundle_name, "filepath": filepath}

    # THIS FUNCTION IS SOLELY WRITTEN BY CLAUDE, NEED TO CLEANUP AND IMPROVE IT
    def _add_bundle_visualizations(self, bundle_data: Dict[str, str]) -> None:
        """
        Add bundle visualization section to the MultiQC report using Niivue interactive viewer.

        Parameters
        ----------
        bundle_data : Dict[str, str]
            A dictionary mapping bundle names to their file paths.
        """

        # Filter for CC bundles only
        cc_bundles = {k: v for k, v in bundle_data.items() if k.startswith("CC")}

        if not cc_bundles:
            log.warning("No CC bundles found for visualization")
            return

        # Define a color palette for different bundles (RGB 0-255)
        color_palette = [
            [255, 0, 0, 255],  # Red
            [0, 255, 0, 255],  # Green
            [0, 128, 255, 255],  # Blue
            [255, 255, 0, 255],  # Yellow
            [255, 0, 255, 255],  # Magenta
            [0, 255, 255, 255],  # Cyan
            [255, 128, 0, 255],  # Orange
            [128, 0, 255, 255],  # Purple
            [0, 255, 128, 255],  # Spring Green
            [255, 192, 203, 255],  # Pink
        ]

        # Prepare bundle data for Niivue JavaScript viewer
        bundles_for_niivue = []

        # Get the analysis directory - it may be a list, so take the first one
        # Prepare bundle metadata (names and colors only - no file embedding)
        for idx, (bundle_name, filepath) in enumerate(sorted(cc_bundles.items())):
            bundles_for_niivue.append({"name": bundle_name, "color": color_palette[idx % len(color_palette)]})
            log.info(f"Prepared bundle metadata for {bundle_name}")

        # Generate the Niivue HTML content directly
        import json

        bundle_json = json.dumps(bundles_for_niivue)
        bundle_count = len(bundles_for_niivue)

        niivue_html = (
            """
<div class="niivue-bundle-viewer">
    <div class="alert alert-info" style="margin-bottom: 20px;">
        <h5>📁 Load Bundle Files</h5>
        <p>Select the .trk bundle files from your file system to visualize them in 3D:</p>
        <input type="file" id="bundle-file-input" multiple accept=".trk" class="form-control" style="max-width: 500px;" />
        <small class="text-muted">Expected bundles: """
            + ", ".join(sorted(cc_bundles.keys()))
            + """</small>
    </div>
    
    <div class="niivue-canvas-container">
        <div id="niivue-loading" class="niivue-loading">Select bundle files above to begin visualization</div>
        <canvas id="niivue-canvas" width="1200" height="600" style="background: #000;"></canvas>
    </div>
    
    <div class="niivue-controls" style="margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
        <h4>Bundle Controls</h4>
        
        <div class="control-group" style="margin: 10px 0;">
            <label>View Mode:</label>
            <button class="btn btn-sm btn-outline-secondary" onclick="window.nvBundles.setSliceType(window.nvBundles.sliceTypeRender)">3D Only</button>
            <button class="btn btn-sm btn-outline-secondary" onclick="window.nvBundles.scene.renderAzimuth = 180; window.nvBundles.scene.renderElevation = 0; window.nvBundles.updateGLVolume()">Superior View</button>
            <button class="btn btn-sm btn-outline-secondary" onclick="window.nvBundles.scene.renderAzimuth = 90; window.nvBundles.scene.renderElevation = 0; window.nvBundles.updateGLVolume()">Lateral View</button>
        </div>
        
        <div class="control-group" style="margin: 10px 0;">
            <label for="fiber-opacity">Bundle Opacity:</label>
            <input type="range" id="fiber-opacity" min="0" max="1" step="0.1" value="1.0" 
                   oninput="updateBundleOpacity(this.value)">
            <span id="opacity-value">1.0</span>
        </div>
        
        <div class="control-group" style="margin: 10px 0;">
            <label for="fiber-color">Fiber Coloring:</label>
            <select id="fiber-color" class="form-select form-select-sm" style="display: inline-block; width: auto;" onchange="updateFiberColor(this.value)">
                <option value="DPV0">Embedded Colors</option>
                <option value="Global">Global Direction</option>
                <option value="Local">Local Direction</option>
                <option value="solid">Solid Color</option>
            </select>
        </div>
        
        <div class="control-group" style="margin: 10px 0;">
            <label>Available Bundles ("""
            + str(bundle_count)
            + """ total):</label>
            <div class="bundle-list" id="bundle-list" style="display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px;">
            </div>
        </div>
    </div>
</div>

<script type="module">
    import { Niivue } from 'https://unpkg.com/@niivue/niivue@0.57.0/dist/index.js';
    
    const bundleData = """
            + bundle_json
            + """;
    let nvBundles;
    let activeBundles = new Set();
    let bundlesLoaded = false;
    
    // Set up file input handler
    document.getElementById('bundle-file-input').addEventListener('change', handleFileSelection);
    
    async function handleFileSelection(event) {
        const files = event.target.files;
        if (files.length === 0) return;
        
        console.log('Files selected:', files.length);
        document.getElementById('niivue-loading').textContent = 'Loading ' + files.length + ' bundle files...';
        document.getElementById('niivue-loading').style.display = 'block';
        
        if (!bundlesLoaded) {
            await initNiivueViewer();
        }
        
        await loadBundlesFromFiles(files);
    }
    
    async function loadBundlesFromFiles(files) {
        console.log('=== Starting Bundle Loading from Files ===');
        let loadedCount = 0;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            try {
                console.log(`[${i+1}/${files.length}] Loading:`, file.name);
                
                // Read file as ArrayBuffer
                const arrayBuffer = await file.arrayBuffer();
                console.log('  File loaded:', arrayBuffer.byteLength, 'bytes');
                
                // Extract bundle name from filename
                const bundleName = file.name.replace('_labels_uniformized.trk', '').split('__')[1] || file.name;
                
                // Load the mesh into Niivue
                const mesh = await nvBundles.loadFromArrayBuffer(arrayBuffer, file.name);
                console.log('  Mesh loaded:', mesh);
                
                if (mesh && mesh.id !== undefined) {
                    // Find matching bundle color
                    const bundleInfo = bundleData.find(b => b.name === bundleName);
                    if (bundleInfo) {
                        nvBundles.setMeshProperty(mesh.id, 'rgba255', bundleInfo.color);
                    }
                    
                    nvBundles.setMeshProperty(mesh.id, 'fiberColor', 'DPV0');
                    nvBundles.setMeshProperty(mesh.id, 'fiberRadius', 0.5);
                    nvBundles.setMeshProperty(mesh.id, 'fiberDither', 0.2);
                    activeBundles.add(mesh.id);
                    
                    console.log('  ✓ Successfully loaded:', bundleName);
                    loadedCount++;
                } else {
                    console.warn('  ⚠ Mesh loaded but no ID returned');
                }
            } catch (error) {
                console.error(`  ✗ Failed to load ${file.name}:`, error);
            }
        }
        
        console.log(`=== Loading Complete: ${loadedCount}/${files.length} bundles loaded ===`);
        
        createBundleList();
        document.getElementById('niivue-loading').style.display = 'none';
        bundlesLoaded = true;
    }
    
    async function initNiivueViewer() {
        console.log('=== Niivue Initialization Started ===');
        console.log('Niivue constructor available:', typeof Niivue);
        
        try {
            console.log('Creating Niivue instance...');
            nvBundles = new Niivue({
                backColor: [0, 0, 0, 1],
                show3Dcrosshair: false,
                dragMode: 2,
            });
            console.log('Niivue instance created:', nvBundles);
            
            console.log('Attaching to canvas...');
            await nvBundles.attachTo('niivue-canvas');
            console.log('Attached to canvas successfully');
            
            console.log('Setting slice type...');
            nvBundles.setSliceType(nvBundles.sliceTypeRender);
            nvBundles.scene.renderAzimuth = 180;
            nvBundles.scene.renderElevation = 0;
            console.log('View configured');
            
            window.nvBundles = nvBundles;
            
            console.log('=== Niivue Initialization Complete ===');
            console.log('Number of bundles available:', bundleData.length);
            console.log('Bundle data:', bundleData);
        } catch (error) {
            console.error('=== Niivue Initialization Failed ===');
            console.error('Error:', error);
            document.getElementById('niivue-loading').textContent = 'Failed to initialize Niivue viewer. Check console for details.';
            document.getElementById('niivue-loading').style.color = 'red';
        }
    }
    
    function createBundleList() {
        const listContainer = document.getElementById('bundle-list');
        listContainer.innerHTML = '';
        
        for (let mesh of nvBundles.meshes) {
            const button = document.createElement('button');
            button.className = 'btn btn-sm btn-success';
            button.textContent = mesh.name || `Bundle ${nvBundles.meshes.indexOf(mesh) + 1}`;
            button.dataset.meshId = mesh.id;
            
            button.onclick = function() {
                if (activeBundles.has(mesh.id)) {
                    nvBundles.setMeshProperty(mesh.id, 'visible', false);
                    activeBundles.delete(mesh.id);
                    button.classList.remove('btn-success');
                    button.classList.add('btn-outline-secondary');
                } else {
                    nvBundles.setMeshProperty(mesh.id, 'visible', true);
                    activeBundles.add(mesh.id);
                    button.classList.remove('btn-outline-secondary');
                    button.classList.add('btn-success');
                }
                nvBundles.updateGLVolume();
            };
            
            listContainer.appendChild(button);
        }
    }
    
    window.updateBundleOpacity = function(value) {
        document.getElementById('opacity-value').textContent = value;
        for (let mesh of nvBundles.meshes) {
            nvBundles.setMeshProperty(mesh.id, 'opacity', parseFloat(value));
        }
        nvBundles.updateGLVolume();
    };
    
    window.updateFiberColor = function(colorMode) {
        for (let mesh of nvBundles.meshes) {
            nvBundles.setMeshProperty(mesh.id, 'fiberColor', colorMode);
        }
        nvBundles.updateGLVolume();
    };
    
    window.addEventListener('DOMContentLoaded', initNiivueViewer);
</script>
"""
        )

        # Add section with Niivue viewer
        self.add_section(
            name="Bundle Visualizations",
            anchor="bundle_visualizations",
            description="Interactive 3D visualization of corpus callosum (CC) bundles. "
            "Use your mouse to rotate, zoom, and explore the fiber tracts in 3D.",
            content=niivue_html,
        )
