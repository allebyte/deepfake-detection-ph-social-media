# Comparative Analysis of AI-Driven Models for Detecting Deepfake Videos in the Philippine Social Media

**Undergraduate Thesis — Bachelor of Science in Computer Science**  
College of Computer Studies, San Pablo Colleges, San Pablo City, Laguna  

* **Researchers:** Ellaine B. Emralino & Nica Joy Catipon
* **Thesis Adviser:** Prof. Solejaena D. Herrera, MIT

---

## Overview
This repository contains the experimental source code, Google Colab notebook, evaluation metrics, statistical test outputs, and 5-fold training logs for a controlled comparative evaluation of five AI-driven deepfake detection architectures across global benchmark datasets (`dev_720` and `test_180` under clean, moderate, and heavy compression tiers) and a supplementary Philippine social media evaluation set (`ph_in_the_wild`).

### Evaluated Architectures
* **CNN-Based Models:** `EfficientNet-B0`, `Xception`
* **Hybrid Spatial-Temporal Model:** `CNN+LSTM`
* **Transformer-Based Models:** `GenConViT-ED`, `FakeFormer-S`

### Programming Language and Tools
Python, PyTorch, OpenCV (YuNet), Scikit-learn, Pandas, FFmpeg, FFprobe, and Google Colab

---

## Repository Structure
* `notebooks/` — End-to-end Google Colab experimentation, training, and evaluation notebook(s)
* `src/` — Modular Python preprocessing scripts (uniform 16-frame sampling & YuNet facial ROI extraction)
* `results/` — Summary tables, per-fold metrics, video-level predictions, statistical test outputs, and evaluation plots
* `logs/` — Epoch-by-epoch training and validation history logs across all 5 folds (`25` training runs)

---

## Quick-Access Artifacts (CS Validation & Thesis Deliverables)

### 1. Detection Performance & Efficiency Tables (SOP 1, SOP 2 & SOP 4)
* [test180_summary_table.csv](results/test180_summary_table.csv) — Global benchmark detection metrics across Clean, Tier 1 (Moderate), and Tier 2 (Heavy) conditions
* [ph_summary_table.csv](results/ph_summary_table.csv) — Cross-domain evaluation results on the Philippine social media dataset (`ph_in_the_wild`)
* [sop4_computational_efficiency.csv](results/sop4_computational_efficiency.csv) — GPU inference latency (ms/video), checkpoint size (MB), parameter counts, and peak CUDA VRAM
* [ph_pairs_preview.jpg](results/ph_pairs_preview.jpg) — Visual verification contact sheet of the 10 identity-matched Philippine video pairs

### 2. Statistical Significance Outputs (SOP 3)
* [stats_cochrans_q.csv](results/stats_cochrans_q.csv) & [stats_mcnemar_pairwise.csv](results/stats_mcnemar_pairwise.csv) — Omnibus Cochran's *Q* and Holm-Bonferroni corrected pairwise McNemar's tests on `test_180`
* [stats_friedman_all_metrics.csv](results/stats_friedman_all_metrics.csv) — Friedman's test across 5 cross-validation folds (Accuracy, Precision, Recall, F1-score, and ROC-AUC)
* [stats_ph_cochrans_q.csv](results/stats_ph_cochrans_q.csv) & [stats_ph_mcnemar_pairwise.csv](results/stats_ph_mcnemar_pairwise.csv) — Non-parametric tests on `ph_in_the_wild`

### 3. Evaluation Plots & Training Logs
* **Figures:** [fig_degradation_robustness.png](results/fig_degradation_robustness.png) | [fig_roc_curves_test180.png](results/fig_roc_curves_test180.png) | [fig_roc_curves_ph.png](results/fig_roc_curves_ph.png)
* **5-Fold Training Histories:** [logs/](logs/)

> **Note on Large Binary Files:** Raw `.mp4` video datasets (`dev_720`, `test_180`, `ph_in_the_wild`), extracted facial frame crops, and trained `.pth` model checkpoints are archived externally on Google Drive due to GitHub file-size limits.
