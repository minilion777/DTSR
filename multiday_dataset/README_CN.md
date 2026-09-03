# 最简多日场景数据集

本目录包含论文运行所需的 680 个预处理场景：500 个训练场景、60 个验证场景和 120 个测试场景。

每个场景仅保留两项最小运行输入：

- `data/<split>/vehicles/*.csv`：车辆到达时隙、停留时长和站点；
- `data/<split>/signals/*.json`：96 个 15 分钟时隙的 WT、PV、负荷和电价。

`manifests/paired_scenario_manifest.csv` 固定每个场景的车辆文件和信号文件配对，并固定 train/val/test 划分。所有攻击、Clean 和防御条件必须在同一个测试场景索引上运行。

最小读取示例：

```python
from integration.multiday_runtime import PairedScenarioDataset

dataset = PairedScenarioDataset("multiday_dataset", split="train", seed=42)
episode = dataset.load_episode(0)
arrivals = episode["arrivals"]
signal_path = episode["signal_path"]
```

原始会话来源、许可和处理边界见根目录 `DATA_LICENSE.md`。
