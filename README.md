# DTSR: Decision-aware temporal state repair for deep reinforcement learning-based electric vehicle charging scheduling under observation attacks

## Overview

This repository provides the implementation and experiment pipeline for DTSR, a temporal state-repair framework for robust deep reinforcement learning-based electric vehicle charging scheduling under observation attacks. It includes the charging environment, training and evaluation code, attack implementations, and the dataset splits used in the experiments.

### Attack settings

The attack suite is organized by temporal scope:

- **Short-horizon attacks:** Opposite-PGD, Opposite-FGSM, Q-function, and ElectHacker-C/F/O perturb individual decision steps. They test immediate policy stability by reversing the intended action direction, reducing the critic-estimated value, or manipulating information related to charging cost, future electricity prices, and vehicle departure.
- **Long-horizon attacks:** Local Small-Drift Q and Local Deadline-Drift PGD accumulate low-amplitude perturbations across a vehicle's parking trajectory. Small-Drift Q repeatedly follows a value-degrading direction, whereas Deadline-Drift PGD introduces departure-stage-aware drift that increases the risk of departure SOC violations.

### DTSR defense

DTSR combines four complementary defense modules:

- **Denoiser (DAE):** uses temporal observation history to reconstruct a candidate clean state and primarily repairs abrupt or localized perturbations.
- **Posterior Benefit Detector (DET):** estimates whether the candidate reconstruction is likely to improve the downstream scheduling decision, applying it selectively to avoid unnecessary distortion of clean observations.
- **Temporal Shield:** uses the previous defended state, executed action, and known system dynamics to construct a one-step feasible region for core variables and projects temporally inconsistent observations back into that region.
- **Persistent-Divergence Belief Correction (PD-BCR):** targets low-amplitude long-horizon drift that can remain locally plausible. It compares the shielded trajectory with a separately propagated vehicle-level belief and applies bounded belief fusion when the divergence remains persistent.

The first three modules address pointwise corruption and local temporal inconsistency, while PD-BCR complements them by suppressing persistent drift accumulated over longer horizons.

## Data source

The experimental scenarios are derived from the official [EV Charge Station Use (September 2018–August 2019)](https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8) dataset published by Perth & Kinross Council, together with experiment-specific operating signals.

The repository contains 680 multi-day scenarios, divided into 500 training scenarios, 60 validation scenarios, and 120 test scenarios.

## Installation

Create and activate a Python virtual environment, then install the required libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the code

Run the complete training and evaluation pipeline:

```powershell
python run_pipeline.py --device auto --seed 42
```

Generated models, intermediate files, and evaluation results are written to the `runs/` directory.

## Repository structure

- `multiday_dataset/`: training, validation, and test scenarios.
- `evc/`: charging environment, scheduling policy, attacks, and DTSR implementation.
- `configs/`: experiment configuration files.
- `scripts/`: training and evaluation entry points.
- `tests/`: lightweight correctness tests.
- `run_pipeline.py`: end-to-end training and evaluation entry point.
- `requirements.txt`: Python dependencies.
