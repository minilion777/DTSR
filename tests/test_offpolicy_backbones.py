import numpy as np
import torch

from evc.merged_core import ReplayBuffer
from evc.merged_core import Actor, Critic
from evc.offpolicy_backbones import (
    GaussianActor,
    MinTwinCritic,
    SACAgent,
    TD3Agent,
    load_evaluation_backbone,
)
from evc.ppo_backbone import PPOAgent, StateValueNetwork, ValueAsQAdapter


def _batch(device: torch.device) -> dict[str, torch.Tensor]:
    replay = ReplayBuffer(64, 11, 1, device)
    for index in range(32):
        obs = np.full(11, index / 100.0, dtype=np.float32)
        replay.add(obs, obs + 0.01, np.array([0.1], dtype=np.float32), -1.0, False)
    return replay.sample(16)


def test_gaussian_actor_forward_is_deterministic_and_bounded():
    actor = GaussianActor()
    obs = torch.randn(8, 11)
    first = actor(obs)
    second = actor.mean_action(obs)
    assert torch.allclose(first, second)
    assert first.shape == (8, 1)
    assert bool(torch.all(first <= 1.0))
    assert bool(torch.all(first >= -1.0))


def test_td3_update_and_attack_critic_contract():
    device = torch.device("cpu")
    agent = TD3Agent(device)
    metrics = agent.update(_batch(device))
    critic = agent.attack_critic()
    value = critic(torch.randn(4, 11), torch.zeros(4, 1))
    assert isinstance(critic, MinTwinCritic)
    assert value.shape == (4, 1)
    assert np.isfinite(metrics["critic1_loss"])


def test_sac_update_and_deterministic_runtime_contract():
    device = torch.device("cpu")
    agent = SACAgent(device)
    metrics = agent.update(_batch(device))
    obs = torch.randn(4, 11)
    assert agent.actor(obs).shape == (4, 1)
    assert agent.act_tensor(obs, explore=True).shape == (4, 1)
    assert np.isfinite(metrics["actor_loss"])
    assert metrics["alpha"] > 0.0


def test_load_evaluation_backbone_supports_ddpg(tmp_path):
    path = tmp_path / "ddpg.pt"
    torch.save(
        {
            "model_type": "baseline_bundle",
            "actor_state_dict": Actor().state_dict(),
            "critic_state_dict": Critic().state_dict(),
            "metadata": {"checkpoint_episode": 1},
        },
        path,
    )
    actor, critic, payload = load_evaluation_backbone("ddpg", path, torch.device("cpu"))
    assert actor(torch.zeros(2, 11)).shape == (2, 1)
    assert critic(torch.zeros(2, 11), torch.zeros(2, 1)).shape == (2, 1)
    assert payload["algorithm"] == "ddpg"


def test_ppo_update_and_value_adapter_contract():
    device = torch.device("cpu")
    agent = PPOAgent(device, update_epochs=2, minibatch_size=8)
    observations = np.random.randn(32, 11).astype(np.float32)
    with torch.no_grad():
        actions = agent.act_tensor(torch.as_tensor(observations), explore=True).cpu().numpy()
    metrics = agent.update(
        {
            "observations": observations,
            "next_observations": observations + 0.01,
            "actions": actions,
            "rewards": np.full(32, -0.1, dtype=np.float32),
            "dones": np.zeros(32, dtype=np.float32),
        }
    )
    critic = agent.attack_critic()
    values = critic(torch.as_tensor(observations[:4]), torch.as_tensor(actions[:4]))
    assert isinstance(agent.value, StateValueNetwork)
    assert isinstance(critic, ValueAsQAdapter)
    assert values.shape == (4, 1)
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert agent.actor(torch.as_tensor(observations[:4])).shape == (4, 1)
