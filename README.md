# DTSR: Decision-aware temporal state repair for deep reinforcement learning-based electric vehicle charging scheduling under observation attacks

## Overview

This repository provides the implementation and experiment pipeline for DTSR, a temporal state-repair framework for robust deep reinforcement learning-based electric vehicle charging scheduling under observation attacks. It includes the charging environment, training and evaluation code, attack implementations, and the dataset splits used in the experiments.

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
