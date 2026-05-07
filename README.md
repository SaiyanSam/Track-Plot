# 🚗 3DGS Vehicle Trajectory Reconstruction for ICVGIP

## 📌 Project Goal

This project aims to estimate robust **3D trajectories of moving vehicles** using camera poses and object detections, and integrate these trajectories into a reconstructed **3D Gaussian Splatting (3DGS)** scene.

The long-term objective is to develop a **temporally consistent trajectory recovery framework** for dynamic objects inside neural-rendered environments and submit the work to **ICVGIP**.

---

# 🎯 Core Objectives

- Recover vehicle trajectories in global 3D coordinates
- Use camera geometry + tracking for localization
- Integrate trajectories into 3DGS scenes
- Improve robustness under:
  - noisy detections
  - occlusions
  - missing tracks
- Develop a publishable research contribution

---

# 📚 Key Research Areas

- 3D Gaussian Splatting (3DGS)
- Dynamic Neural Rendering
- Multi-view Geometry
- Object Tracking
- Temporal Trajectory Optimization

---

# 🗓️ 8-Week Research Plan

| Week | Phase | Tasks | Status |
|------|------|------|--------|
| **Week 1** | **Pipeline Cleanup + Literature Survey** | - Rebuild clean modular pipeline from reference scripts<br>- Read key papers (3DGS, Dynamic NeRF/GS, tracking, triangulation)<br>- Write literature notes + identify research gaps<br>- Standardize configs, outputs, and folder structure<br>- Verify end-to-end trajectory → 3DGS overlay works cleanly | 🚧 In Progress |
| **Week 2** | **Strong Baseline Construction** | - Implement proper triangulation baseline<br>- Implement DeepSORT/ByteTrack + geometry baseline<br>- Implement homography-based baseline (current best)<br>- Create unified trajectory export format<br>- Select datasets (custom + optional Wildtrack/KITTI) | Planned |
| **Week 3** | **Core Contribution Design** | - Decide main novelty direction:<br>&nbsp;&nbsp;• temporal optimization<br>&nbsp;&nbsp;• uncertainty-aware smoothing<br>&nbsp;&nbsp;• trajectory-ground consistency<br>&nbsp;&nbsp;• dynamic-object-aware 3DGS integration<br>- Formulate equations<br>- Implement proposed method v1<br>- Generate initial comparison results | Planned |
| **Week 4** | **Robustness + Temporal Modeling** | - Handle missing detections and ID switches<br>- Improve trajectory stability under occlusion<br>- Add temporal filtering / optimization<br>- Add uncertainty estimation or confidence weighting<br>- Begin quantitative evaluation metrics | Planned |
| **Week 5** | **Experiments + Evaluation** | - Run all baselines vs proposed method<br>- Generate RMSE / reprojection / smoothness metrics<br>- Run ablation studies<br>- Generate qualitative visualizations<br>- Create 3DGS trajectory overlays for paper figures | Planned |
| **Week 6** | **Paper Writing + Final Experiments** | - Write Introduction + Related Work<br>- Write Method section with diagrams/equations<br>- Write Experimental Setup<br>- Finalize remaining experiments<br>- Prepare tables and plots | Planned |
| **Week 7** | **Paper Refinement + Visual Quality** | - Improve figures and visualizations<br>- Add failure cases and limitations<br>- Improve trajectory rendering quality<br>- Refine contribution framing<br>- Complete full paper draft | Planned |
| **Week 8** | **Submission Preparation** | - Format in ICVGIP template<br>- Proofread and reduce ambiguity<br>- Finalize references and captions<br>- External feedback/review if possible<br>- Final submission | Planned |

---

# 📊 Planned Evaluation Metrics

- RMSE trajectory error
- Reprojection error
- Temporal smoothness
- Occlusion robustness
- Trajectory stability
- Qualitative 3DGS overlay quality

---

# 📂 Planned Repository Structure

```bash
project_root/
│
├── configs/
├── scripts/
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   └── ...
│
├── src/
│   ├── tracking/
│   ├── geometry/
│   ├── smoothing/
│   ├── visualization/
│   └── evaluation/
│
├── results/
├── outputs/
├── logs/
├── docs/
└── README.md
```

---

# 📖 Literature Survey Papers

## Core 3DGS
- 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## Dynamic Scene Modeling
- D-NeRF
- Dynamic 3D Gaussians

## Tracking
- DeepSORT
- ByteTrack

## Geometry
- Multi-View Triangulation
- Structure from Motion (SfM)

---

# 🚀 Current Focus

## Week 1 Goals
- Clean modular pipeline
- Literature survey
- Stable trajectory extraction
- Reliable 3DGS integration
- Research gap identification

---

# 🧠 Current Research Direction

> Temporally consistent 3D trajectory recovery and integration inside neural-rendered 3DGS scenes.

---

# 📌 Target Conference

## ICVGIP 2026
Goal: Submission-ready paper with:
- clear novelty
- strong baselines
- quantitative evaluation
- compelling visualizations
