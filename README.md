Nerfstudio's command is just a wrapper, so it needs the actual COLMAP engine installed on your Linux machine to do the math. Since you are using Conda, this is incredibly easy:  
	conda install -c conda-forge colmap  
	git clone --recursive https://github.com/cvg/Hierarchical-Localization.git  
	cd Hierarchical-Localization  
	pip install -e .  
	cd ..  
	
To estimate the intrinsics and extrincs:  
	ns-process-data video --data data/demo1.mp4 --output-dir results/colmap_output --sfm-tool hloc --feature-type superpoint --matcher-type superpoint+lightglue
To train from the above genearted output:
	ns-train splatfacto --data results/colmap_output
Go to browser:
	http://localhost:7007

# 🚗 3DGS Trajectory Reconstruction Project (ICVGIP Submission)

## 📌 Problem Statement
We aim to estimate the **3D trajectory of a moving object (car)** using multi-view detections and integrate this trajectory into a **3D Gaussian Splatting (3DGS)** reconstructed scene. The goal is to move beyond simple visualization and develop a **robust, temporally consistent trajectory estimation method** that works under noisy detections and real-world constraints.

---

## 🎯 Objectives
- Build a complete pipeline: detection → tracking → triangulation → 3D trajectory → 3DGS overlay  
- Develop a **novel method** for improving trajectory estimation (e.g., temporal smoothing / optimization / joint modeling)  
- Compare against strong baselines  
- Produce a **publishable paper for ICVGIP (8 weeks timeline)**  

---

## 📚 Core References
- 3D Gaussian Splatting (Kerbl et al., 2023)  
- D-NeRF (Dynamic NeRF)  
- DeepSORT / ByteTrack (Tracking)  
- Multi-view triangulation (geometry baseline)  

---

## 🗓️ 8-Week Plan

| Week | Phase | Tasks | Status |
|------|------|------|--------|
| **Week 1** | **Foundation + Literature Survey** | - Read key papers (3DGS, D-NeRF, tracking, triangulation)<br>- Write 1–2 page literature notes<br>- Define problem statement + contribution idea<br>- Fix existing pipeline bugs (end-to-end working) | Planned |
| **Week 2** | **Baseline Setup** | - Implement triangulation-only baseline<br>- Implement tracking (DeepSORT/ByteTrack) + triangulation<br>- Prepare dataset (Wildtrack/custom)<br>- Standardize pipeline I/O | Planned |
| **Week 3** | **Method Design (Core Contribution)** | - Choose contribution (temporal smoothing / optimization / joint modeling)<br>- Formulate equations<br>- Implement initial version (v1)<br>- Verify basic improvements over baseline | Planned |
| **Week 4** | **Method Improvement** | - Handle noise, occlusions, missing detections<br>- Improve stability of trajectory<br>- Debug and refine method<br>- Start defining evaluation metrics (RMSE, smoothness, reprojection error) | Planned |
| **Week 5** | **Experiments + Evaluation** | - Run comparisons (Baseline 1 vs Baseline 2 vs Proposed)<br>- Generate quantitative metrics<br>- Create visualizations (trajectory + 3DGS overlay)<br>- Perform ablation study | Planned |
| **Week 6** | **Paper Writing (Draft)** | - Write Introduction + Related Work<br>- Write Method section (with diagrams)<br>- Organize experiments and figures<br>- Prepare 60–70% draft | Planned |
| **Week 7** | **Results + Refinement** | - Improve figures and visual quality<br>- Write Results + Discussion sections<br>- Add failure cases and limitations<br>- Complete full paper draft | Planned |
| **Week 8** | **Final Polish + Submission** | - Format paper (ICVGIP template)<br>- Proofread and refine writing<br>- Finalize references and figures<br>- External review (if possible)<br>- Submit paper | Planned |

---

## 📊 Evaluation Plan
- **Trajectory Error (RMSE / ATE)**  
- **Temporal Smoothness**  
- **Reprojection Error**  
- **Qualitative Visualization in 3DGS**  

---

## 🧠 Key Contribution Goal
> Develop a method for **robust and temporally consistent 3D trajectory estimation integrated with neural rendering (3DGS)**.

---

## ⚠️ Important Notes
- Focus on **clear contribution**, not just pipeline  
- Strong **baselines + comparisons** are critical  
- Visual results (3DGS + trajectory) are **very important for acceptance**  

---

## ✅ Progress Tracker Legend
- `Planned` → Not started  
- `In Progress` → Currently working  
- `Done` → Completed  

---

## 🚀 Final Goal
Submission-ready **ICVGIP paper** with:
- Clear novelty  
- Strong experiments  
- High-quality visuals  
