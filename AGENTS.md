# BW Residual RL Repository Instructions

本文件适用于整个仓库。任何自动化代理或协作者在修改代码前，都必须先阅读本文件和
`docs/PROJECT_CONTEXT.md`。子目录若以后新增更具体的 `AGENTS.md`，以离目标文件最近的规则为准，
但不得放宽本文的机器人安全要求和数据/模型兼容性约束。

## 开始工作前

1. 阅读 `docs/PROJECT_CONTEXT.md`，确认改动属于采集、推理还是训练链路。
2. 运行 `git status --short`，保留用户已有改动；不要回滚、覆盖或顺手格式化无关文件。
3. 先读目标模块的 README、`configs/default.yaml` 或 CLI `--help`，再改实现。
4. 涉及真实机器人时，先判定任务是离线代码工作、只读诊断、dry-run，还是实际控制。
5. 只做请求范围内的改动。不要提交数据集、模型权重、缓存、日志或 ROS 运行产物。
6. 如果没有要求修改代码，就先不要修改代码，只有在用户告诉，进行修改的时候再改
7. `docs/PROJECT_CONTEXT.md`的最后一栏是项目进展，所有对话如果涉及到项目的实验、代码修改等相关进展，必须在`docs/PROJECT_CONTEXT.md`文档的最后一栏更新

## 项目目标与目录

本项目实现 BW 双臂机器人的完整离线学习链路：采集 BC 示教、训练 ACT、采集人工纠正、训练
Residual BC、采集 rollout、训练 Offline Residual SAC/CQL，并部署 ACT + residual 策略。

| 目录 | 职责 | 主要入口 |
| --- | --- | --- |
| `lerobot_bw_data_collector/` | 从 ROS2 话题写入 BC/RL LeRobot 数据集，并检查、合并数据 | `scripts/collect.sh`、`scripts/check_topics.sh`、`scripts/check_dataset.sh` |
| `lerobot_bw_policy_runner/` | 加载 ACT/Residual checkpoint，组装观测并发布控制与 debug 动作 | `scripts/check_inputs.sh`、`scripts/run_infer.sh` |
| `lerobot_bw_rl/` | ACT 视觉缓存、Residual BC、Offline Residual SAC/CQL 训练及奖励检查 | `scripts/build_visual_cache.sh`、`scripts/train_bc.sh`、`scripts/train_sac.sh` |

完整操作顺序和示例参数见 `BW训练全流程操作手册.md`。模块细节分别见三个子项目的 README。

机器人的ROS控制系统的代码，在主目录下的：~/bw_teleoperate_ws，这个目录下有本课题的基本思路文档，以及接口文档，这些文档必要时需要读取。
机器人采集的数据在：~/robot_datasets，这里包含了采集的数据，必要是可以读取。

## 运行环境

当前目标环境是 Python 3.10、ROS 2 Humble、LeRobot 0.4.4，默认虚拟环境和 ROS workspace 如下。
这些初始化片段按 Bash 语法执行；当前终端若是 zsh，先进入 `bash` 子 shell，或改用对应的
`setup.zsh`：

```bash
source ~/venvs/lerobot_ros310/bin/activate
source /opt/ros/humble/setup.bash
source ~/bw_teleoperate_ws/install/setup.bash
```

仓库脚本会自动加载这些环境，正常运行优先使用 `scripts/*.sh`，不要在文档或自动化中复制一套
不同的启动逻辑。默认机器人序列号为 `BW_IZN3E0FU`，但代码不得把 CLI 已支持的覆盖能力删除。

三个 `pyproject.toml` 是独立 Python 子项目。不要假设仓库根目录存在统一构建系统，也不要随意升级
LeRobot、Torch、ROS 或 checkpoint 格式；这类升级需要单独的兼容性任务和真实资产验证。

## 不可破坏的接口契约

### 相机合同 v3

- 顺序固定为 `env_cam`、`left_wrist_cam`、`right_wrist_cam`。
- 环境 D435：`640x480`、`rgb8`、30 FPS。
- 左右 D405：`480x270`、`rgb8`、30 FPS。
- 数据与策略输入使用 RGB `uint8` HWC；当前 `IMAGE_TRANSFORM` 为 `none_exact_shape`，不在链路中静默缩放。
- 每路最低稳定帧率为 28.5 FPS，最大帧龄 0.15 秒。正式采集和推理要求每轮取得三路唯一新帧。

相机常量在 collector、runner 和 RL 训练代码中都有对应定义。改变名称、顺序、尺寸、编码、帧率或
变换时，必须同步修改三处实现、配置、checkpoint 元数据校验、测试和文档。不要为了让旧数据或旧
checkpoint 通过而弱化校验。

### 16 维关节合同

固定顺序是左臂 7 关节、左夹爪、右臂 7 关节、右夹爪。权威名称见：

- `lerobot_bw_data_collector/src/lerobot_bw_data_collector/constants.py`
- `lerobot_bw_policy_runner/src/lerobot_bw_policy_runner/constants.py`

`observation.state`、ACT 动作、执行动作和 debug 动作均使用该 16 维顺序。Residual actor 只回归
14 维手臂 residual；夹爪索引 7 和 15 使用离散三分类控制。修改关节名、别名、维度或顺序时，必须
同时检查 joint mapping、数据 schema、训练输入、动作组合、可视化和 checkpoint 加载。

### 数据与 checkpoint 合同

- BC 与 RL 数据 schema 不同，禁止合并到同一数据集。
- RL 数据中的 `action` 等于实际执行动作；`control_source=0` 表示人工，`1` 表示策略。
- Residual BC/RL 必须绑定生成视觉特征时使用的同一个 ACT checkpoint。ACT SHA256 指纹不匹配应立即失败。
- Residual BC 与从其初始化的 RL 必须保持视觉定义、输入维度、`hidden_dims`、归一化、lambda、limits
  和夹爪标签语义一致。
- 当前 residual checkpoint `format_version=4`，旧的连续夹爪 residual checkpoint 不兼容。
- `lerobot_bw_rl/configs/*.yaml` 仅是说明示例；训练 CLI 参数才是权威输入。

## 机器人安全规则

- 诊断任务默认只读：可以查看进程、GPU、ROS graph/topic rate、日志和内核 USB 信息，不发布动作、
  不切换控制源、不重启节点、不拔插设备。
- 未经用户明确要求，不启动真实控制。实际部署前必须先检查输入，再运行有限步 dry-run：

```bash
cd lerobot_bw_policy_runner
./scripts/check_inputs.sh --robot-sn BW_IZN3E0FU --timeout 10
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path /path/to/pretrained_model \
  --device cuda \
  --fps 30 \
  --dry-run \
  --max-steps 30
```

- `--dry-run` 不发布控制或 debug 消息；不要移除或绕过此保证。
- 任一路相机停滞时，严格的新帧门控应停止推理和动作发布。这是防止陈旧图像控制的安全机制，
  不能把关闭 `require_new_frames` 当作卡顿修复。
- 物理重连相机、USB 线或更换端口前，必须停止推理或切回人工控制。
- 实际发布前校验 ACT/residual 指纹、相机合同和输入 shape；失败时应在发布第一条动作前退出。

## 修改原则

- 以既有模块边界和脚本入口为准，保持 Python 3.10 兼容。
- 配置、CLI、README 与行为必须一致。新增参数时同步更新 dataclass/解析、默认值、shell 入口、测试和文档。
- 对外字段或 ROS topic 变更属于跨模块接口变更，先搜索所有生产者、消费者和测试。
- collector 与 runner 中存在有意重复的相机转换、相机状态和关节映射逻辑。修改其中一份时必须比较另一份，
  并保持同一合同下的行为一致。
- 保留严格、显式的错误。数据字段缺失、shape 不符、相机帧率不足或 checkpoint 不兼容时，不做静默填充、
  resize 或降级。
- 不编辑生成目录：`outputs/`、`checkpoints/`、数据集、`.bw_act_visual_cache/`、日志和视频均不属于源码。
- 不加入真实机器路径、序列号以外的敏感信息、私有数据样本或大文件。

## 验证要求

至少运行与改动模块对应的测试。以下命令从仓库根目录的 Bash 环境执行。当前机器需要禁止自动加载
不兼容的第三方 pytest 插件，并且
`PYTHONPATH` 必须保留 ROS setup 已加入的路径；三个测试目录应分开运行，以避免同名测试模块冲突。

```bash
source ~/venvs/lerobot_ros310/bin/activate
source /opt/ros/humble/setup.bash
source ~/bw_teleoperate_ws/install/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/lerobot_bw_data_collector/src:$PYTHONPATH" \
python -m pytest -q lerobot_bw_data_collector/tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/lerobot_bw_policy_runner/src:$PYTHONPATH" \
python -m pytest -q lerobot_bw_policy_runner/tests

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/lerobot_bw_rl:$PYTHONPATH" \
python -m pytest -q lerobot_bw_rl/tests
```

验证范围按风险扩展：

| 改动 | 最低验证 |
| --- | --- |
| 单模块纯逻辑 | 对应模块测试 |
| 相机、图像、关节或数据 schema | 三个测试集 + 数据检查工具的相关模式 |
| 训练/checkpoint 格式 | RL 与 runner 测试，并做 checkpoint 加载或最小合成 smoke test |
| ROS topic 或发布语义 | collector + runner 测试；有硬件时先 `check_inputs.sh` 和 dry-run |
| Shell 脚本 | `bash -n <script>`，再验证非破坏性参数或 `--help` |

真实机器人、相机、GPU checkpoint 或大型数据集不可用时，不要伪称完成了端到端验证；明确写出未验证部分。

## 卡顿与低频诊断顺序

策略进程存在不等于控制健康，瞬时接近 30 Hz 也不等于稳定。按以下顺序收集证据：

1. 确认 runner PID、ROS node、GPU 利用率和最近日志。
2. 分别测量三路相机 topic 的持续帧率、帧龄和重复帧。
3. 查看 inference 窗口、`camera_wait_cycles` 和 `last_wait_reason` 对应逻辑。
4. 确认 `control_source`：`0=REMOTE`，`1=INFERENCE`。
5. 确认 policy action 正在发布，并与变化中的 joint feedback 对照。
6. 检查 RealSense 日志及内核 USB/UVC 错误，再判断是相机/USB、ROS、模型吞吐还是控制源问题。

优先修复输入链路或增加节流诊断日志，不要先放宽相机门控。进行上述检查时保持只读。

## 文档维护

- 架构、合同、topic、模式或安全流程改变时，同步更新 `docs/PROJECT_CONTEXT.md`。
- 用户操作步骤或训练参数改变时，同步更新 `BW训练全流程操作手册.md` 和对应模块 README。
- 文档中的路径和参数应标明是合同、默认值还是示例；不要把本机某次运行的 PID、速率或日志结论写成永久事实。
