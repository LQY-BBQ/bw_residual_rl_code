# BW Residual RL 项目上下文

更新时间：2026-08-05

本文描述当前仓库的稳定架构和跨模块合同，供开发者与自动化代理快速建立共同上下文。它不是一次训练
实验记录，也不替代完整操作手册。修改代码前还应阅读根目录 `AGENTS.md` 和目标模块 README。

## 1. 项目要解决的问题

BW 双臂机器人先通过人工示教训练 ACT 基础策略，再在执行 ACT 时采集人工接管纠正，用 Residual BC
学习局部修正；随后采集成功与失败 rollout，以 Offline Residual SAC/CQL 继续训练。RL 更新完全离线，
不会在控制机器人时在线更新模型。

```text
ROS2 状态 + 三路 RGB 相机
            |
            v
       BC 示教数据 -> ACT 训练 -----------------------+
            |                                        |
            v                                        v
ACT rollout + 人工接管 -> RL correction 数据 -> Residual BC
                                                    |
                                                    v
                                ACT + Residual BC rollout
                                                    |
                                                    v
                                      Offline Residual SAC/CQL
                                                    |
                                                    v
                                        ACT + Residual RL 部署
```

基础 ACT 始终冻结。Residual 策略复用同一次 ACT forward 中的视觉特征，不为 residual 额外执行第二次
图像预处理或视觉 backbone。

## 2. 仓库结构与所有权

### `lerobot_bw_data_collector`

负责 ROS2 到 LeRobot 数据集的边界：

- `src/.../collect.py`：采集主循环和 episode 结束语义。
- `src/.../ros_reader.py`：订阅状态、人工动作、相机及 policy debug 话题，组装同步样本。
- `src/.../dataset_writer.py`：定义 BC/RL features 并写帧。
- `src/.../keyboard_marker.py`：采集时写 reward/done/success。
- `tools/check_bw_lerobot_dataset.py`：检查 schema、图像和动作关系。
- `scripts/merge_lerobot_datasets.sh`：只重新编号并合并兼容数据，不改动作、奖励或图像。

### `lerobot_bw_policy_runner`

负责模型到 ROS2 控制接口的边界：

- `src/.../infer_node.py`：启动检查、30 Hz 主循环、ACT/residual 组合和发布。
- `src/.../ros_io.py`：读取完整观测、执行相机新帧门控、发布控制/debug topic。
- `src/.../policy_loader.py`：加载并校验 ACT，执行 ACT forward 和共享视觉特征提取。
- `src/.../residual_policy.py`：加载 format v4 residual checkpoint 并校验绑定关系。
- `src/.../action_utils.py`：residual 组合、clamp、平滑和 ROS `JointState` 转换。
- `src/.../gripper_control.py`：ACT 夹爪迟滞与 residual 三分类控制。
- `src/.../handover_control.py`：人工接管期间的 Policy shadow 缓存与无冲击恢复状态机。
- `src/.../visualization/`：只读动作可视化。

### `lerobot_bw_rl`

负责离线训练：

- `build_act_visual_cache.py` / `visual_cache.py`：为数据集预计算冻结 ACT 视觉特征。
- `train_residual_bc.py`：训练确定性 14 维手臂 residual + 双夹爪三分类网络。
- `train_residual_sac.py`：训练离线 SAC，可加入 BC regularization 和 CQL penalty。
- `bw_datasets/residual_transition_dataset.py`：从 RL 数据构造监督样本与 transition。
- `policies/act_shared_encoder.py`：定义训练侧 ACT 加载、相机顺序和共享视觉特征。

## 3. 运行环境和外部依赖

当前经过设计的目标组合为：

| 项目 | 当前合同 |
| --- | --- |
| Python | 3.10 或以上；实际部署环境为 3.10 |
| ROS | ROS 2 Humble |
| LeRobot | 0.4.4 |
| 虚拟环境 | `~/venvs/lerobot_ros310` |
| ROS workspace | `~/bw_teleoperate_ws` |
| 默认 robot SN | `BW_IZN3E0FU` |
| ROS domain | 0 |
| 控制/采集频率 | 30 Hz |

底层机器人和相机通常由 `~/bw_teleoperate_ws/scripts/local/start_lqy.sh` 启动。该 workspace、真实相机、
机器人硬件、ACT checkpoint 和数据集都在本仓库之外。

三个 Python 子项目可分别以 editable 方式安装，但日常命令应使用各目录的 shell 入口，因为入口会加载
虚拟环境、ROS Humble 和机器人 workspace。生成的数据、视觉缓存、训练输出及 checkpoint 都被 `.gitignore`
排除，不应进入 Git。

## 4. 相机合同 v3

相机顺序是模型输入的一部分，不只是显示名称。

| 名称 | ROS topic | 源规格 | LeRobot shape |
| --- | --- | --- | --- |
| `env_cam` | `/camera/env_d435/color/image_raw` | D435 `640x480 rgb8` | `(480, 640, 3)` |
| `left_wrist_cam` | `/camera/left_d405/color/image_raw` | D405 `480x270 rgb8` | `(270, 480, 3)` |
| `right_wrist_cam` | `/camera/right_d405/color/image_raw` | D405 `480x270 rgb8` | `(270, 480, 3)` |

稳定性要求：

- 期望 30 FPS，检查下限 28.5 FPS。
- 最大允许帧龄 0.15 秒；collector 的持续停滞阈值为 0.5 秒。
- ROS header 时间戳必须有效；帧率按唯一时间戳计算。
- `require_new_frames=true` 时，每个采集/推理步必须同时获得三路唯一新帧。
- 转换结果为连续内存的 RGB `uint8` HWC，不进行隐式 resize。
- 合同版本为 3，图像变换标识为 `none_exact_shape`。

因此任一路相机停止更新时，采集器不重复写旧帧，runner 也不推理和发布动作。这是有意的安全行为，
外观上可能像推理“卡住”。

## 5. 关节和动作合同

所有状态和完整动作采用固定 16 维顺序：

```text
0  left_shoulder_pitch_joint
1  left_shoulder_yaw_joint
2  left_shoulder_roll_joint
3  left_elbow_joint
4  left_wrist_roll_joint
5  left_wrist_pitch_joint
6  left_wrist_yaw_joint
7  left_gripper_joint
8  right_shoulder_pitch_joint
9  right_shoulder_yaw_joint
10 right_shoulder_roll_joint
11 right_elbow_joint
12 right_wrist_roll_joint
13 right_wrist_pitch_joint
14 right_wrist_yaw_joint
15 right_gripper_joint
```

消息按关节名映射，不依赖 ROS 消息的原始顺序。已知的大小写名称和 elbow/gripper 别名会规范化，
pelvis/head 等非采集关节允许作为额外字段存在。

手臂 residual 的网络输出是归一化 14 维向量。部署时：

```text
delta_joint_arm = delta_normalized_arm * residual_limits_arm
action_composed_arm = action_act_arm + residual_lambda * delta_joint_arm
```

然后按配置执行 clamp 和指数平滑。夹爪不使用连续 residual：Residual BC/RL 为左右夹爪分别输出
`KEEP_BASE`、`FORCE_OPEN`、`FORCE_CLOSE` 三分类，再由确认帧数、置信度、迟滞和最短保持时间生成最终
离散命令。debug `action.rl_delta` 的夹爪维固定为 0。

## 6. ROS topic 边界

默认 SN 为 `BW_IZN3E0FU`；下表中的 `{robot_sn}` 由配置展开。

### Runner 输入

| 语义 | Topic |
| --- | --- |
| 关节反馈 | `/{robot_sn}/joint_states_fdb` |
| 控制源 | `/{robot_sn}/Teleop/control_source` |
| 人工夹爪交接输入（不进入模型） | `/{robot_sn}/Teleop/gripper_pos` |
| 三路图像 | 见相机合同 |

### Runner 控制输出

| 语义 | Topic |
| --- | --- |
| 14 维手臂命令 | `/{robot_sn}/Policy/joint_angle_solution/smooth` |
| 2 维夹爪命令 | `/{robot_sn}/Policy/gripper_pos` |

### Runner debug 输出

| 语义 | Topic |
| --- | --- |
| ACT 完整动作 | `/{robot_sn}/Policy/debug/action_act` |
| joint-space residual | `/{robot_sn}/Policy/debug/action_rl_delta` |
| residual 组合且应用夹爪候选后的动作 | `/{robot_sn}/Policy/debug/action_composed` |
| 实际发布到 Policy 控制入口的最终动作 | `/{robot_sn}/Policy/debug/action_final` |
| 左右夹爪原始 residual 类别 | `/{robot_sn}/Policy/debug/gripper_residual_class` |

debug topic 用于采集和观测，不应由机器人下位机直接执行。runner 的 `--dry-run` 不发布控制和 debug
消息。RL collector 因此必须连接到非 dry-run runner，才能要求并记录完整 debug schema。

人工遥控动作来自：

- `/{robot_sn}/Teleop/joint_angle_solution/smooth`
- `/{robot_sn}/Teleop/gripper_pos`

`control_source=0` 表示 REMOTE/人工控制，`control_source=1` 表示 INFERENCE/策略控制。

Runner 只有收到合法 `control_source` 才允许发布完整 Policy 控制。REMOTE 期间模型继续推理并发布
ACT/residual/composed 诊断，手臂最终缓存则跟随 14 维反馈；Teleop 夹爪只同步交接安全缓存，不进入
模型。切回 INFERENCE 时清空 ACT temporal ensemble 和后处理历史，保持 6 个有效帧，再以
`0.15 rad/s`、单步 `0.005 rad@30Hz` 且命令-反馈不超过 `0.03 rad` 的限制恢复。连续 3 帧距目标不超过
`0.01 rad` 后恢复正常状态。缺少近期合法 Teleop 夹爪时仍保留模型诊断，但不发布控制或
`debug/action_final`。

下游 `bw_serial` 在 REMOTE→INFERENCE 切换点另设时间戳门控：切换前的 Policy 缓存全部拒绝，只有切换
后新产生的手臂和夹爪消息同时到达才完成交接；等待期间保持最后实际使用的 Teleop 命令。

## 7. 推理模式

| CLI mode | ACT | Residual | 用途 |
| --- | --- | --- | --- |
| `act` | 每控制步 fresh forward | 无，delta 为 0 | 基础部署和 correction 采集 |
| `act_residual_bc` | 每控制步 fresh forward | 确定性 BC actor | 纠正策略部署和 RL rollout 采集 |
| `act_residual_rl` | 每控制步 fresh forward | SAC actor，通常 deterministic | 最终离线 RL 部署 |
| `act_residual_sac` | 同上 | 同上 | 仅保留的旧 CLI 别名 |

ACT 的 temporal ensemble 由 checkpoint 配置保留。Residual 模式通过 ACT
`encoder_img_feat_input_proj` 的 forward hook，从同一次 forward 获取三路池化视觉特征。

发布前 runner 会检查：

- ACT 输入相机字段、顺序、shape 和 16 维 state/action。
- residual 类型是否匹配运行 mode。
- ACT checkpoint SHA256 指纹是否与 residual 训练时一致。
- visual dimension、相机合同版本、图像变换、源/策略图像 shape 和数据 FPS。
- residual checkpoint `format_version=4`、归一化参数、lambda 和 limits。

## 8. 数据集语义

### BC schema

```text
observation.state
observation.images.env_cam
observation.images.left_wrist_cam
observation.images.right_wrist_cam
action
```

`action` 是 Teleop 手臂与夹爪组成的 16 维人工动作。

### RL schema

RL 包含全部 BC 观测，并额外保存：

```text
control_source
is_intervention
has_human_action
action.act
action.rl_delta
action.human
action.executed
action.gripper_policy_class
reward
done
success
timing.*
```

关键语义：

- `is_intervention = (control_source == 0)`，当前 `has_human_action` 与其相同。
- 接管帧：`action.executed = action.human`。
- 非接管帧：`action.executed = Policy/debug/action_final`。
- RL 模式的顶层 `action` 始终等于 `action.executed`。
- `reward/done/success` 在采集时直接写入 parquet，不再依赖旧 annotation JSON 流程。
- `Ctrl+C` 或 `--max-frames` 结束时，最后一帧会作为失败终止帧落盘。
- 终止帧会保留进 RL transition；`done=1` 屏蔽 next-Q，避免跨 episode bootstrap。

BC 与 RL features 不同，合并工具应拒绝混合。合并前先 dry-run，合并后用 dataset checker 和
`check_rl_rewards.sh --strict` 检查。

## 9. 训练和 checkpoint 关系

Residual 输入统一为：

```text
[ACT pooled projected visual features, observation.state, action.act]
```

视觉特征定义为三路 ACT preprocessor -> ResNet layer4 -> `encoder_img_feat_input_proj` -> 每路空间
平均池化 -> 按固定相机顺序拼接。ACT 完全冻结。

Residual BC 在人工接管帧学习人工动作与 ACT 动作之差，非接管帧的手臂目标为 0；夹爪使用从完整
episode 动作离散化得到的三分类标签。Residual SAC 只优化 14 维手臂 actor/critic；从 BC 初始化时，
夹爪分类网络作为独立冻结网络保留。

Checkpoint 保存并约束：ACT 指纹、输入/视觉维度、网络尺寸、观测归一化、residual lambda/limits、
相机合同与 shape、数据集 FPS、夹爪标签配置和格式版本。不能用“看起来相同”的另一个 ACT checkpoint
替换；必须使用构建视觉缓存和训练 residual 时的确切 ACT 目录。

`lerobot_bw_rl/configs/residual_bc.yaml` 和 `residual_sac.yaml` 是说明性示例。实际训练配置由 CLI 决定，
完整且较新的命令以 `BW训练全流程操作手册.md` 为准。

## 10. 标准工作流入口

详细参数见完整操作手册，开发和排障优先记住以下入口：

```bash
# 采集前 topic/schema 检查
cd lerobot_bw_data_collector
./scripts/check_topics.sh --robot-sn BW_IZN3E0FU --mode bc
./scripts/check_topics.sh --robot-sn BW_IZN3E0FU --mode rl

# 部署前输入与三路相机持续帧率检查
cd ../lerobot_bw_policy_runner
./scripts/check_inputs.sh --robot-sn BW_IZN3E0FU --timeout 10

# 数据集检查
cd ../lerobot_bw_data_collector
./scripts/check_dataset.sh /path/to/dataset --mode bc --all-episodes
./scripts/check_dataset.sh /path/to/dataset --mode rl --all-episodes --save-csv

# RL reward/terminal 检查
cd ../lerobot_bw_rl
./scripts/check_rl_rewards.sh /path/to/rl_dataset --show-events --strict
```

真实部署必须先 `check_inputs.sh`，再用目标 checkpoint 执行 `run_infer.sh --dry-run --max-steps 30`。
只有输入合同、模型加载和有限步推理都通过，才进入真实发布。

## 11. 已知边界与诊断提示

- 当前实现是离线 Residual BC + Offline SAC，可选 BC regularization 和 CQL；不是在线 RL，也不是完整
  RLPD actor/learner/buffer 系统。
- runner 活着、GPU 有利用率或短时间达到 30 Hz，都不能单独证明控制链路可靠。要同时查看三路相机、
  inference 窗口、动作发布、控制源和反馈跟踪。
- 严格新帧门控会在任一相机停滞时抑制动作。`last_wait_reason` 可区分等待新图和图像过期；诊断时应
  增加可见日志或检查 topic，不应关闭门控。
- 历史排障曾发现左 D405 的 USB/UVC 不稳定导致推理低频。它是历史证据，不代表当前硬件状态；出现
  类似症状时应重新检查 RealSense 日志、内核 USB 错误和每路持续帧率。
- 物理处理相机或 USB 前，必须停止 inference 或切回人工控制。

## 12. 文档权威顺序

出现不一致时按以下层级核对，并修正文档漂移：

1. 运行代码中的显式校验和数据结构。
2. collector/runner 的 `configs/default.yaml` 默认值及 CLI 解析。
3. `BW训练全流程操作手册.md` 的当前完整流程命令。
4. 三个子项目 README 的模块说明。
5. 根目录中带日期的修改报告或旧说明文件，仅作历史参考。

架构、合同或安全流程变更后更新本文的“更新时间”，并同步维护 `AGENTS.md` 中的执行规则。

## 13. 项目进展

2026.7.30
采集了48组数据，训练了机器人。机器人所做的任务是：左手把旁边的方块拿起来放到中间白色的盒子里面，
然后右手把方块拿起来放到中间的盒子里面。有时候还可以放到上一个方块上面。这次采集，我设置了左边方块有
4个位置，右边也有4个。这样组合有16种，所以每种采集了三组。

2026.7.31
进行部署测试，修改了夹爪开合的限制参数，因为有时候会出现左侧第二个点，他不知道要闭合夹爪。除了这个小问题之外，
机器人基本上可以顺利的执行所有的动作。只要是16种组合中的任意一种，机器人都能够顺利执行。
我还测试了另一种情况，就是机器人执行完了之后，再次放置物块，之前是每次都重置动作，这次没有，然后让机器人继续执行，
机器人的执行效果不好。

2026.8.3
重新部署测试，中间机器人经过了位移，摄像头可能也有点歪了，我稍微调整了一下，但是这次的效果还是很差。
主要表现在夹爪有时候不知道要闭合，然后左手放完了物块，就会卡住，右手就不知道要拿物块了，呆呆的。
我不知道目前应该怎么办，我本来打算，今天进行残差BC的训练，但是，目前原模型的效果不太行。我要解决这个问题还是咋办。

2026.8.5
目前发现之前部署的时候效果不好的原因了。是因为机器人的夹爪被换掉了。他的颜色发生了变化。这个是导致效果变差的主要原因。
目前我准备进行残差数据采集，但是我发现，目前有时候方块放在桌子上，一些角度下，机械臂抓取会稍微偏离方块，当方块处于夹爪内部时候，
机械臂会把方块先移动到某个位置，然后再抓取。我觉得可能是因为机器人的位置偏移，导致他抓取这个位置的方块的时候，和采集的数据有偏移，
方块在夹爪里面的位置和采集的数据不匹配，然后，所以他无法识别这个情况，就移动夹爪，直到和常见位置类似，然后才夹取方块。
针对这种情况，我如果切换到人工操控，调整夹爪位置，在切换回去的时候，手臂会立刻产生巨大偏移，因为策略话题保存输出位置与此刻位置偏差大，
所以切换的时候，机械臂会忽然产生巨大移动，这样会破坏之前的状态，还是无法让手臂更好的抓取物块。目前我不知道应该怎么做。我觉得可以采取办法，让
切换回去策略控制的时候，不会有巨大动作。

2026.8.5（人工接管安全交接实现）
已在 policy runner 增加 `REMOTE_SHADOW -> INITIAL_HOLD -> RESUMING -> INFERENCE` 状态机。人工期间手臂
Policy 缓存跟随反馈，恢复时重置 ACT temporal ensemble，保持 6 帧并按 `0.15 rad/s`、`0.03 rad`
跟踪误差上限平滑接管；Teleop 夹爪仅用于交接，缺失或非法不影响模型推理但阻止完整控制发布。
`bw_serial` 同步增加切换后新 Policy 手臂/夹爪双流门控，避免旧缓存重放和半切换。当前仅完成离线
单元测试与隔离构建，尚未进行真实机器人切换验证；实机前仍需先执行输入检查和 30 步 dry-run。

2026.8.5（数据检查手册更新）
Residual BC 纠正数据检查流程改为直接调用 `check_dataset.sh` 检查指定的单个数据集，不再在手册中使用
批量检查脚本；当前示例检查 `act_correction_001` 的全部 episode，并输出到其独立可视化目录。

2026.8.5（27 组 Residual BC 纠正数据检查）
已离线检查 `act_correction_001` 至 `027`，共 47,114 帧、约 26.16 分钟。27 组 dataset checker
均无 `ERROR`，strict reward 检查均通过；16 维动作、相机 v3、parquet/视频帧数、接管动作、夹爪端点和
reward/done/success 合同一致，81 个视频均可完整解码。总接管 8,366 帧，其中策略开始后的实际纠正为
7,594 帧、84 段。`act_correction_027` 只有开头 0-12 帧处于人工控制，策略运行后没有纠正，应人工确认
它是有意保留的零 residual 成功样本，还是应排除/补采。全部数据均缺少 residual lambda 元数据，检查器
按默认 0.2 解读，训练前仍需确认实际 lambda；timing 告警只出现在每组开头 3-5 个 REMOTE 帧，未影响
Policy 控制帧。检查结果位于 `~/robot_datasets/pick_block_to_box/Res_BC_Data_viz/`。检查器现有的 14 维
手臂 target/16 维绘图索引问题仅用进程内适配绕过，本次未修改仓库源码或输入数据。

2026.8.5（本批 Residual BC 训练预检与命令校准）
合并 dry-run 已确认 `act_correction_*` 共 27 episodes、47,114 frames。按训练器的夹爪标签算法统计，
左开/左关、右开/右关独立事件分别为 `11/17`、`18/23`，未达到每类至少 20 次的硬门槛，因此本批尚未
启动训练，最低还需补采左开 9、左关 3、右开 2 个独立事件，推荐四类均补到 25 次。操作手册第六节已
改为当前 `Res_BC_Data` 的 dry-run、正式合并和训练命令，固定绑定 ACT `050000` checkpoint，并根据
本批纠正幅度使用 `residual_lambda=1.0`、手臂 limit `0.20`；训练先运行 20,000 steps 并每 2,000 steps
检查验证集。format v4 夹爪为离散三分类，命令中不再使用已废弃的连续夹爪 residual limit。
