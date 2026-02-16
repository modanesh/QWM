# 🚀 Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning (QWM)

This repository contains the official PyTorch implementation of **Quadrupedal World Models (QWM)**, a framework designed to transition world models from hardware-locked specialists to physics-grounded generalists.

By explicitly conditioning the generative dynamics on a robot's engineering specifications (extracted from USD files), QWM enables a single model to master locomotion across a highly heterogeneous fleet of quadrupeds (e.g., ANYmal, Unitree, Spot). This capability unlocks the ability to deploy zero-shot control on entirely unseen quadrupeds without requiring fine-tuning, adaptation, or dangerous real-world warm-up periods.

## ✨ Key Features

* 🧬 **Physical Morphology Encoder (PME):** Extracts scale-invariant physical parameters (kinematics, geometry, dynamics, and actuation) directly from Unified Robot Description Format (USD) assets.
* 🧠 **Morphology-Conditioned Dynamics:** Employs a dual-tower encoder and injects static structural data directly into the Recurrent State-Space Model (RSSM), relieving the RNN from memorizing static parameters.
* ⚖️ **Adaptive Reward Normalization (ARN):** Dynamically balances highly heterogeneous reward scales across different robot types using quantile-based Exponential Moving Averages (EMA), stabilizing simultaneous multi-robot training.
* 🌍 **Hetero-Isaac Environment:** Built on top of NVIDIA Isaac Lab, supporting parallelized, heterogeneous batches where distinct collision geometries, kinematic trees, and actuator gains are simulated simultaneously.

## 📂 Repository Structure

```text
QWM/
├── assets/
│   └── usds/                       # Contains .usd files for all robots and pre-extracted features
│       ├── anymal_b.usd
│       ├── anymal_c.usd
│       ├── anymal_d.usd
│       ├── spot.usd
│       ├── unitree_a1.usd
│       ├── unitree_b2.usd
│       ├── unitree_go1.usd
│       ├── unitree_go2.usd
│       └── usd_physical_features_minmax_1.npz
├── configs.yaml                      # Hyperparameter configurations and robot parameters
├── qwm.py                        # Main entry point for training and evaluation
├── envs/
│   └── hq_isaac.py                   # Heterogeneous Isaac Lab environment wrapper
├── feature_extractors/             
│   └── manual_extract_embeddings.py  # Script for extracting morphological features from USDs
├── models.py                         # WorldModel and ImagBehavior architectures
├── networks.py                       # Core neural network modules (RSSM, MultiEncoder, MLP, etc.)
├── utils.py                          # Replay buffer, logging, and USDFeatureManager utilities
└── requirements.txt

```

## ⚙️ Installation

> **Note:** This project has been tested with **Isaac Sim 4.5** and **Isaac Lab 2.1.0**.

This project requires a specialized version of Isaac Lab to support heterogeneous quadruped environments (the `HeteroQuadruped` environment), which is not available in the standard Isaac Lab release.

1. **Clone and install the custom Isaac Lab fork:**
```bash
git clone https://github.com/modanesh/IsaacLab-HQ.git
cd IsaacLab-HQ
# Follow the standard Isaac Lab installation instructions in the repo

```


2. **Clone this repository:**
```bash
git clone https://github.com/modanesh/QWM.git
cd QWM

```


3. **Install dependencies:**
Ensure you are using the Python environment provided by your Isaac Sim installation.
```bash
pip install -r requirements.txt

```



## 💻 Usage

### Training

To train the QWM agent across the heterogeneous cohort of quadrupeds, use the Python executable bundled with your Isaac Sim installation.
Then, run the main training script with your desired configuration:
```bash
python_sim qwm.py --configs hetero_quadruped_proprio_exp

```



Configurations can be modified in `configs.yaml`. Logs, model checkpoints, and training metrics will be saved in the generated `logdir/` directory.

## 📝 Citation

If you find this code or research helpful in your work, please cite the paper:

```bibtex
@article{danesh2024qwm,
  title={Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning},
  author={Danesh, Mohamad H. and Li, Chenhao and Abyaneh, Amin and Houssaini, Anas and Ellis, Kirsty and Berseth, Glen and Hutter, Marco and Lin, Hsiu-Chin},
  journal={TBD},
  year={2026}
}

```
