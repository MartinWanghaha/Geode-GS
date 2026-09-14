# Geode-GS: Geometrically-Guided Dense 3D Gaussian Splatting

<p align="center">
    <img src="https://github.com/MartinWanghaha/Geode-GS/raw/main/assets/pre.png?raw=true" alt="Geode-GS Overview" width="100%">
</p>
<p align="center">
    <em>Overview of the Geode-GS framework. Our method integrates dense geometric initialization, a geometry-aware rendering pipeline, and fused normal supervision to achieve high-fidelity novel view synthesis and accurate surface reconstruction.</em>
</p>

## 📜 Introduction

3D Gaussian Splatting (3DGS) enables real-time, high-quality novel view synthesis through differentiable rasterization. However, its optimization is driven entirely by photometric losses on 2D images, which encourages the model to find “shortcut” solutions that fit the training views rather than recover physically correct geometry. This inherent limitation often leads to **surface artifacts**, **floaters**, and **holes in the reconstructed geometry**.

To address these issues, we propose **Geode-GS**, a framework for **geometrically-guided dense 3D Gaussian Splatting**. Our method combines **geometric fidelity** and **photorealism** within a unified optimization objective. Our contributions include:

- **📍 Dense Geometric Initialization:** We use dense feature matching and triangulation to generate an overcomplete set of geometrically accurate Gaussian primitives with global scene coverage in a single initialization stage, directly addressing the sparsity of the initial point cloud.
- **📐 Geometry-Aware Rendering:** We design a rendering pipeline that produces per-pixel depth and surface normals, providing explicit spatial information for geometric supervision.
- **💡 Fused Normal Supervision:** We introduce a novel fused normal supervision strategy. Filtering guided by normal priors produces high-quality, robust fused normal maps, which provide an additional geometric loss and address the limitations of photometric supervision alone.

Experiments show that Geode-GS achieves state-of-the-art rendering quality on several challenging benchmarks, including Mip-NeRF 360, Deep Blending, and Tanks & Temples. More importantly, our method substantially improves **geometric accuracy**, enabling the extraction of **clean, high-fidelity surface meshes**. The reconstructed objects can thus serve as fully functional geometric assets for downstream applications such as scene composition and editing.

<p align="center">
    <img src="assets/guocheng.png" alt="Geode-GS Pipeline Visualization" width="100%">
    <em>Visualization of our geometry-aware rendering pipeline. From left to right and top to bottom: ground-truth image, rendered image, rendered normals, depth normals, estimated depth, depth-derived normals, fused normals, and edge weights.</em>
</p>

## ✨ Results

### Qualitative Rendering Comparison

Compared with current state-of-the-art methods such as EDGS and PGSR, Geode-GS produces renderings with noticeably fewer artifacts and greater detail fidelity.

![Qualitative Comparison](./assets/duibi.png)

### Robustness from Extreme Viewpoints

When viewed from novel and challenging viewpoints, Geode-GS preserves scene structure and background completeness, whereas other methods exhibit severe floaters or geometric distortions.

![Extreme Viewpoints](./assets/jiduan.png)

### High-Fidelity Mesh Extraction

The high geometric accuracy of Geode-GS enables the extraction of clean, detailed surface meshes, extending 3DGS from a rendering representation to a source of fully functional geometric assets.

![Mesh Extraction](./assets/mesh.png)

### Scene Composition and Editing

Assets reconstructed from entirely different capture environments, such as benches, bicycles, and tables, can be freely combined and arranged to create a new, coherent composite scene.

![Scene Composition](./assets/3dm.png)

## 🚀 Getting Started

### 1. Environment Setup

First, clone this repository and its submodules:

```bash
git clone --recursive https://github.com/MartinWanghaha/Geode-GS.git
cd Geode-GS
```

We recommend creating a virtual environment with Conda:

```bash
conda create -n geode-gs python=3.9
conda activate geode-gs
```

Then, install the required dependencies. We depend on the `diff-gaussian-rasterization` and `simple-knn` libraries used in the original 3DGS implementation.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install the Gaussian rasterization submodule
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..

# Install the simple-knn submodule
cd submodules/simple-knn
pip install .
cd ../..
```

### 2. Data Preparation

Our method follows the same data structure as the original 3DGS. Process your dataset with **COLMAP** to obtain camera poses. The processed dataset should have the following structure:

```text
<scene_path>
├── input/
│   ├── image1.png
│   ├── image2.png
