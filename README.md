# DTSR: Robust State Defense for EV Charging Scheduling

This repository implements DTSR for multi-day electric-vehicle charging scheduling. A DDPG scheduling policy is trained under adversarial state perturbations and protected by four DTSR components: DAE, DeT, Temporal Shield, and UG-BCR.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Run the complete experiment from scratch:

```powershell
python run_pipeline.py --device auto --seed 42
```

The pipeline trains DDPG, collects clean trajectories, trains and calibrates the four DTSR modules, and evaluates short- and long-horizon attacks. Generated checkpoints and evaluation files are written to `runs/`.

## Repository structure

- `multiday_dataset/`: 680 multi-day scenarios (500 training, 60 validation, and 120 test scenarios).
- `evc/`: charging environment, policy training, attacks, and DTSR defense modules.
- `scripts/`: entry points for training and evaluation stages.
- `run_pipeline.py`: end-to-end experiment entry point.

## Data source

The scenarios are derived from the official [EV Charge Station Use (September 2018–August 2019)](https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8) dataset published by Perth & Kinross Council, together with experiment-specific operating signals.
