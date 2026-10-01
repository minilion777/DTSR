# DTSR: Robust State Defense for EV Charging Scheduling

This repository implements DTSR for multi-day electric-vehicle charging scheduling. A DDPG scheduling policy is protected by four components: DAE, DeT, Temporal Shield, and paper-aligned PD-BCR.

The runtime order is:

```text
DAE/DeT routing -> Temporal Shield -> constraint-gated PD-BCR -> Actor
```

PD-BCR uses three conditions described in the paper: sufficiently large belief divergence, persistent divergence direction, and a bounded Shield residual. The belief correction is activated only when all three conditions are satisfied; otherwise, the actor receives the Temporal Shield state unchanged.

The constraint-gate structure follows the paper. The numerical values below are this repository's validation-set defaults, not values claimed to be specified by the paper. They must be frozen before test-set evaluation.

| Parameter | Default |
| --- | ---: |
| Core weights `W_c` | `(1.0, 1.0, 0.1)` |
| Divergence threshold `delta_on` | `0.008` |
| Persistence threshold `rho_p` | `0.0` |
| Shield residual limit `r_max` | `0.06` |
| Persistence window `K` | `3` |
| Fusion weight `lambda` | `0.7` |

The tracked reference configuration is `configs/pd_bcr_config.json`; training writes the frozen copy to `runs/dtsr/pd_bcr_config.json` for evaluation.

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

The unified attack evaluation contains the six short-horizon conditions used by the paper (Opposite-PGD, Opposite-FGSM, Q-function, and ElectHacker-C/F/O) plus the two retained stateful long-horizon attacks.

Run the lightweight correctness tests before experiments:

```powershell
python -m unittest discover -s tests -v
```

## Repository structure

- `multiday_dataset/`: 680 multi-day scenarios (500 training, 60 validation, and 120 test scenarios).
- `evc/pd_bcr.py`: paper-aligned constraint-gated post-Shield PD-BCR implementation.
- `evc/attacks/`: unified short- and long-horizon attack implementations.
- `configs/pd_bcr_config.json`: reviewable reference values for the frozen constraint gate.
- `evc/`: charging environment, policy training, attacks, and the remaining DTSR modules.
- `scripts/`: entry points for training and evaluation stages.
- `run_pipeline.py`: end-to-end experiment entry point.

## Data source

The scenarios are derived from the official [EV Charge Station Use (September 2018–August 2019)](https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8) dataset published by Perth & Kinross Council, together with experiment-specific operating signals.
