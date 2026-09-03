# 多日车辆—外部信号配套数据说明

## 1. 数据规模

- 训练场景：500个
- 验证场景：60个
- 测试场景：120个
- 每个场景：344辆车
- 每天：96个15分钟时隙

## 2. 场景组成

每个场景由三部分组成：

1. `vehicles/*.csv`
   - `Arrive_time`
   - `Duration_of_stay`
   - `Station`

2. `signals/*.json`
   - `WT`
   - `PV`
   - `L`
   - `price`

3. `contexts/*.csv`
   - 96个时隙的外生上下文
   - 新到车辆数
   - 当前在站车辆数
   - 当前PV、WT、基础负荷和电价
   - 未来5个时隙归一化电价

## 3. 数据之间的关系

车辆数据与外部信号没有被强行绑定为同一个真实自然日，但按照以下规则进行合理配对：

- 训练、验证和测试划分完全隔离；
- 季节标签一致；
- 工作日/周末类型一致；
- 在站车辆活跃度只作为负荷和电价的弱行为相关项；
- EV实际充放电功率不写入基础负荷，避免重复计算。

## 4. 生成信号与基准曲线的相关性

全部680个场景的平均相关系数：

- WT：0.9306
- PV：0.9673
- 负荷：0.9259
- 电价：0.9172
- 四类信号总体平均：0.9353

每个单独场景还设置了最低相关性约束，避免产生明显异常的曲线。

## 5. 智能体实际输入

当前项目中的智能体不是每15分钟只接收一个全局向量，而是对每辆正在站内的车辆分别做一次动作决策。

每辆车在时隙 t 的11维输入为：

```text
[
  SOC_i(t),
  remaining_duration_i(t) / 12,
  PV(t) / 100,
  WT(t) / 100,
  Load(t) / 100,
  normalized_price(t+1),
  normalized_price(t+2),
  normalized_price(t+3),
  normalized_price(t+4),
  normalized_price(t+5),
  normalized_cumulative_cost_i(t)
]
```

其中：

- `SOC`和累计成本由环境按照前序动作实时更新；
- 剩余停留时间由到达时长逐时隙递减；
- PV、WT、负荷和未来电价来自配套信号文件；
- `Station`不进入当前11维状态，只用于站点分组和功率统计；
- 当前原始电价用于计算本时隙成本和奖励，但当前状态中使用的是未来5步电价。

因此，`contexts/*.csv`是固定外生部分，完整11维状态仍需在环境运行时组装。

## 6. 接入方法

把整个文件夹放在项目根目录下，然后：

```python
from integration.multiday_runtime import PairedScenarioDataset

dataset = PairedScenarioDataset(
    "perth_multiday_runtime_ready",
    split="train",
    seed=42,
    shuffle=True,
)

scenario = dataset.load_episode(0)
arrivals = scenario["arrivals"]
signal_path = scenario["signal_path"]
context = scenario["context"]
```

多日训练可直接使用：

```python
from integration.multiday_training_adapter import train_agent_multiday

agent, history = train_agent_multiday(
    dataset_root="perth_multiday_runtime_ready",
    device=device,
    episodes=500,
)
```

测试时，应固定120组车辆和信号组合。Clean、所有攻击和所有防御必须在同一个测试索引下共享完全相同的场景。
