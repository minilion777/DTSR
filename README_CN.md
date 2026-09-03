# DTSR：面向 DRL 电动汽车调度观测攻击的离线决策感知时序状态修复

## 数据与目录

- `multiday_dataset/`：完整运行时数据，共 680 个配对场景（500 train、60 val、120 test）与 10 个清单文件。
- `data/`：基线训练使用的参考会话与信号数据。
- `evc/`、`cli.py`：环境、攻击、DTSR/DeT、UG-BCR 和在线防御的核心实现。
- `scripts/`：论文实验的训练、校准、评估与绘图脚本。
- `config/`：攻击、多日 DDPG 与自适应攻击配置。
- `tests/`：单元和回归测试。

## 安装

建议使用 Python 3.10 或 3.11，并在独立虚拟环境中安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU 可用时，`--device auto` 会自动使用 CUDA；没有 CUDA 时可传入 `--device cpu`。

## 从零训练 DTSR

下列入口不依赖仓库内的预训练模型。它依次训练参考 baseline、多日 DDPG、收集 clean trajectories、训练 DTSR/DeT，并默认进行 20 场景评估：

```powershell
python run_pipeline.py --device auto --seed 42
```

## 论文实验对应代码

- 攻击效果与时序特征：`06_evaluate_120_attacks.py`、`16_profile_attack_temporal_features_seed42.py`。
- 防御强度与消融：`17_evaluate_exp2_strength_addendum_seed42.py` 与 `_strength_eval_common.py`。
- UG-BCR / DET：`18_evaluate_exp4_attack_ratio_seed42.py` 至 `20_plot_gate_quality_figures.py`、`25_calibrate_ug_bcr_v3_seed42.py`。
- 模块感知与因果知识阶梯攻击：`21_evaluate_exp4_adaptive_long_horizon_seed42.py` 至 `23_plot_exp4_causal_knowledge_ladder.py`。
- DDPG、TD3、SAC、PPO 跨骨干实验：`26_train_cross_backbone.py` 至 `37_train_ppo_backbone.py`。
- 在线基线对比：`23_evaluate_online_vs_dtsr_default_attacks.py` 与 `cli.py` 的 `train-sa-ddpg`、`train-wocar`、`train-atla-ddpg` 等子命令。

各脚本均支持 `--help`。运行论文评估前，先完成 `run_pipeline.py` 的训练阶段，再通过脚本参数把 actor、bundle、clean 数据和 DTSR 目录指向 `runs/` 下对应文件；不要使用已删除的 `models/`、`artifacts/` 或 `results/` 历史路径。

## 数据来源与许可

场景使用 Perth & Kinross Council 发布的 **EV Charge Station Use (September 2018–August 2019)** 匿名充电会话数据，并结合本文的处理流程和半合成 WT/PV、负荷、价格信号生成。上游数据适用 [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)，来源页见 [ArcGIS Open Data](https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8)。详见 [DATA_LICENSE.md](DATA_LICENSE.md)。

代码采用 MIT License。若需要上游原始会话数据、生成细节或数据问题说明，请通过本仓库 GitHub Issues 联系维护者。
