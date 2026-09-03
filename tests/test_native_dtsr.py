from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from evc.native_dtsr import (
    build_frozen_attacker,
    load_frozen_attack_plan,
    load_frozen_backbone,
    validate_attack_plan_provenance,
    validate_dataset_backbone,
)
from evc.offpolicy_backbones import SACAgent, TD3Agent, save_backbone_bundle
from evc.ug_bcr import load_ug_bcr_config


def _selection() -> dict:
    return {
        "opposite_pgd": {
            "profile_id": "op10_r1",
            "profile": {
                "kind": "pointwise",
                "epsilon": 0.1,
                "alpha": 0.01,
                "iters": 10,
                "restarts": 1,
            },
        },
        "q_function": {
            "profile_id": "qmin20_r2",
            "profile": {
                "kind": "pointwise",
                "epsilon": 0.1,
                "alpha": 0.01,
                "iters": 20,
                "restarts": 2,
                "q_mode": "min",
            },
        },
        "local_small_drift_q": {
            "profile_id": "small_actor_pressure",
            "profile": {
                "kind": "long_horizon",
                "attack_overrides": {"epsilon": 0.055},
            },
        },
        "local_deadline_drift_pgd": {
            "profile_id": "deadline_moderate",
            "profile": {
                "kind": "long_horizon",
                "attack_overrides": {"epsilon": 0.055},
            },
        },
    }


def _write_config(path, *, scenario_id: str = "val_day_0001") -> None:
    path.write_text(
        json.dumps(
            {
                "calibration_split": "val",
                "calibration_scenario_ids": [scenario_id],
                "selected": {"td3": _selection(), "sac": _selection()},
            }
        ),
        encoding="utf-8",
    )


def test_sac_backbone_contract_uses_deterministic_mean(tmp_path) -> None:
    bundle_path = tmp_path / "sac.pt"
    save_backbone_bundle(SACAgent(torch.device("cpu")), bundle_path)
    backbone = load_frozen_backbone("sac", bundle_path, torch.device("cpu"))
    obs = torch.randn(4, 11)
    assert backbone.policy_action_mode == "tanh_mean"
    assert torch.equal(backbone.actor(obs), backbone.actor(obs))
    assert all(not parameter.requires_grad for parameter in backbone.actor.parameters())


def test_frozen_plan_provenance_and_pointwise_budget(tmp_path) -> None:
    bundle_path = tmp_path / "td3.pt"
    config_path = tmp_path / "native.json"
    save_backbone_bundle(TD3Agent(torch.device("cpu")), bundle_path)
    _write_config(config_path)
    backbone = load_frozen_backbone("td3", bundle_path, torch.device("cpu"))
    plan = load_frozen_attack_plan("td3", config_path)

    validate_dataset_backbone(
        {"split": "train", "backbone": backbone.provenance()},
        backbone,
        split="train",
    )
    validate_attack_plan_provenance({"attack_plan": plan.provenance()}, plan)
    attacker = build_frozen_attacker(
        "opposite_pgd",
        backbone=backbone,
        attack_plan=plan,
        device=torch.device("cpu"),
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        seed=42,
    )
    clean = np.full((3, 11), 0.5, dtype=np.float32)
    adversarial = attacker.attack(clean)
    assert adversarial.shape == clean.shape
    assert float(np.max(np.abs(adversarial - clean))) <= 0.1 + 1e-6


def test_frozen_plan_rejects_test_calibration_and_wrong_budget(tmp_path) -> None:
    config_path = tmp_path / "native.json"
    _write_config(config_path, scenario_id="test_day_0001")
    with pytest.raises(ValueError, match="only val"):
        load_frozen_attack_plan("td3", config_path)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["calibration_scenario_ids"] = ["val_day_0001"]
    payload["selected"]["td3"]["q_function"]["profile"]["epsilon"] = 0.2
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="epsilon mismatch"):
        load_frozen_attack_plan("td3", config_path)


def test_ug_bcr_loader_enforces_no_leakage_contract(tmp_path) -> None:
    path = tmp_path / "ug_bcr.json"
    payload = {
        "schema_version": 2,
        "leakage_policy": "strict_no_clean_state",
        "time_initialization": "routed_observation",
        "uses_clean_state": False,
        "uses_true_remaining_time": False,
        "belief": {"time_initialization": "routed_observation"},
        "urgency_gate": {"enabled": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_ug_bcr_config(path)
    assert config.schema_version == 2
    assert config.urgency_gate.enabled

    payload["uses_clean_state"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot use clean state"):
        load_ug_bcr_config(path)
