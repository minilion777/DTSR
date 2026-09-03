# DTSR：EV 充电调度的鲁棒状态防御

本项目复现 DTSR 在多日 EV 充电调度中的实验流程：训练 DDPG 调度策略，在对抗状态扰动下依次使用 DAE、DeT、Temporal Shield 和 UG-BCR 进行防御。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行

从零运行完整流程：

```powershell
python run_pipeline.py --device auto --seed 42
```

流程会训练 DDPG，收集正常轨迹，训练/校准 DTSR 四个模块，并评估两种短时序攻击和两种长时序攻击。生成的模型与评估文件写入 `runs/`。

## 目录

- `multiday_dataset/`：680 个多日场景（500 训练、60 验证、120 测试）。
- `evc/`：充电环境、策略训练、攻击和 DTSR 防御模块。
- `scripts/`：各阶段的训练和评估入口。
- `run_pipeline.py`：完整实验总入口。

## 数据来源

场景基于 Perth & Kinross Council 发布的官方 [EV Charge Station Use (September 2018–August 2019)](https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8) 数据构建，并配套论文实验使用的运行信号。数据许可与署名说明见 `DATA_LICENSE.md`。
