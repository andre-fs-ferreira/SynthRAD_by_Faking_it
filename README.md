# VS-DDPM: 3D Variable-Step Denoising Diffusion Probabilistic Model

## Solution for the SynthRAD2025 Challenge

> **Note:** This project was developed and adapted from the [BraTS 2025 solution](https://github.com/andre-fs-ferreira/BraTS2025_by_Faking_it).

---

### 📌 Overview

Our approach introduces a dynamic Variable-Step mechanism that takes the size of each 3D volume into consideration to adjust the number of inference steps. To achieve this, the model is trained to optimize the pipeline so it can seamlessly operate across a varying number of *T* steps.

---

### ⚠️ Important: Data Pre-processing

The provided dataset is not completely registered out of the box. **You must register the data before training any networks.** Please follow the official instructions here: [SynthRAD2025 Pre-processing Repository](https://github.com/SynthRAD2025/preprocessing).

---

### 🧠 Supported Architectures

We extensively tested various architectures for this pipeline. Currently, only the following models are recommended, as others did not yield better performance:
* **SwinVIT**
* **U-Net**

---

### 📂 Repository Structure

Use the provided shell scripts in the repository to execute training and inference processes. The core Python scripts powering the pipeline are:

* `MonaiDataLoader.py`: Handles data loading and dataset management.
* `train_mc_IDDPM.py`: Main script for model training.
* `infer_mc_IDDPM.py`: Main script for running inference.

---

### 🚀 Training Workflow

Training is strictly divided into a two-stage process to ensure stability and optimal metric performance.

#### Stage 1: Stabilization
In the first step, the goal is to stabilize the high variance initially predicted by the network. 
* **Loss Function:** MAE only.
* **Required Flag:** Add `--penalize_high_variance` to your training script execution.

#### Stage 2: Metric Optimization
Once the network is stable, the variance penalty is removed, and specific training metrics are introduced to refine the model.
* **Action:** Remove the `--penalize_high_variance` flag.
* **Required Flags:** Add `--add_train_metric AFP` and `--add_train_metric_weight 0.2`.

---

### 💬 Support

Let me know if something is not clear, please open an issue!