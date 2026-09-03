# 数据来源、处理边界与许可

## 上游来源

`multiday_dataset/` 中的运行时场景以 Perth & Kinross Council 发布的 **EV Charge Station Use (September 2018–August 2019)** 匿名充电会话为基础。原始发布页为 <https://www.arcgis.com/home/item.html?id=ca6cae3df2624832a2eaf678f2eabee8>。

该上游数据按 [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) 发布。再利用时应保留来源署名、许可链接，并不得暗示数据提供方认可本项目。

建议署名：

> Contains information from Perth & Kinross Council, licensed under the Open Government Licence v3.0.

## 仓库内数据的含义

本仓库发布的是论文运行所需的 680 个预处理配对场景，而不是上游未处理会话数据的镜像：

- 500 个训练场景、60 个验证场景和 120 个测试场景；
- 每个场景包含按实验接口规范化的车辆到达/停留/站点信息；
- 配套的 WT/PV、负荷和价格信号为论文流程中的半合成运行信号；
- `manifests/` 固化了划分与车辆/信号文件配对关系。

因此，仓库可直接复现代码的最小运行输入，但不宣称可仅凭本仓库重新生成与上游逐记录完全一致的原始会话数据。数据读取接口见 `multiday_dataset/README_CN.md` 和 `scripts/02_collect_multiday_clean.py`。

## 联系方式

需要原始数据访问、场景生成细节或发现数据问题时，请通过本仓库 GitHub Issues 联系维护者。代码本身由根目录的 MIT License 许可；上游数据及其衍生数据的权利与条件不因该代码许可而改变。
