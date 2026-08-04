# lerobot_bw_data_collector: BC + Residual RL 数据采集

这个目录负责把 BW 机器人 ROS2 话题采集成 LeRobot 数据集。当前支持两种模式：

- `--mode bc`: 采集普通行为克隆数据，保存状态、三路图像和人工动作。
- `--mode rl`: 采集 ACT rollout + 人工接管/correction 数据，额外保存 ACT 动作、residual delta、最终执行动作、接管标记和键盘 reward。

当前 reward/done/success 是采集时直接写进 parquet 的字段，不再依赖 `annotations/episode_xxxxxx.json`。

## 目录角色

```text
lerobot_bw_data_collector/
  configs/default.yaml          # ROS topic、数据集路径、fps、mode 默认配置
  scripts/collect.sh            # 推荐采集入口，会 source venv/ROS/workspace
  scripts/check_topics.sh       # 采集前检查 ROS topic 和一帧样本能否组装
  scripts/check_dataset.sh      # 采集后检查 LeRobot 数据集字段、图像、动作关系
  src/lerobot_bw_data_collector/
    collect.py                  # 主采集循环
    ros_reader.py               # ROS 订阅和一帧 CollectorSample 组装
    dataset_writer.py           # LeRobot features 和 frame 字典构造
    keyboard_marker.py          # RL 模式键盘 reward/done/success
    joint_mapping.py            # 按关节名映射到固定 16 维顺序
    image_utils.py              # ROS Image -> RGB uint8 HWC
```

## 采集前检查

BC 模式检查：

```bash
./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc
```

RL 模式检查：

```bash
./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl
```

`check_topics.sh` 会检查配置中的 topic、消息类型和图像契约，并用 2 秒窗口统计 ROS header 时间戳去重后的帧率。任一路低于 `28.5 FPS`、图像超过 `0.15s` 未更新、header 时间戳为零或规格不匹配都会失败。

当前三路相机已经按同时稳定 30 FPS 的最高实测配置适配：

| 数据集相机名 | ROS topic | ROS 输入 | 数据集 RGB shape |
| --- | --- | --- | --- |
| `env_cam` | `/camera/env_d435/color/image_raw` | 环境 D435 `640x480`、`rgb8` | `(480, 640, 3)` |
| `left_wrist_cam` | `/camera/left_d405/color/image_raw` | 左 D405 `480x270`、`rgb8` | `(270, 480, 3)` |
| `right_wrist_cam` | `/camera/right_d405/color/image_raw` | 右 D405 `480x270`、`rgb8` | `(270, 480, 3)` |

采集器会读取 ROS `Image.step` 并移除行填充。正式采集只在三路相机都出现新 header 时间戳时写帧，不会把旧图重复保存为新的 30 FPS 数据；任一路持续 `0.5s` 没有新帧会终止采集。建议每次正式采集前运行一次 `check_topics.sh`。

## BC 模式使用

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc \
  --dataset-root ~/robot_datasets/bw_lerobot \
  --session-name bc_demo_001 \
  --task "pick block demo"
```

BC 模式字段：

```text
observation.state
observation.images.env_cam
observation.images.left_wrist_cam
observation.images.right_wrist_cam
action
```

字段来源和处理：

| 字段 | 来源 | 处理 |
| --- | --- | --- |
| `observation.state` | `/{robot_sn}/joint_states_fdb` | 按关节名取 `position`，转 `float32`，重排成固定 16 维；不做归一化、缩放、滤波或单位转换 |
| `action` | `/{robot_sn}/Teleop/joint_angle_solution/smooth` + `/{robot_sn}/Teleop/gripper_pos` | 手臂 14 维 + 夹爪 2 维，按固定 16 维顺序拼接；不做归一化、缩放、滤波或单位转换 |
| `observation.images.*` | 三路 `sensor_msgs/Image` | D435 和 D405 的 `rgb8` 转成连续内存的 RGB `uint8` HWC；底层转换器也能解析 `bgr8/rgba8/bgra8/mono8/yuv422_yuy2` |

## RL 模式使用

RL 采集需要先启动 `lerobot_bw_policy_runner`，因为采集器要读取它发布的 debug 话题：

```text
/{robot_sn}/Policy/debug/action_act
/{robot_sn}/Policy/debug/action_rl_delta
/{robot_sn}/Policy/debug/action_final
/{robot_sn}/Policy/debug/gripper_residual_class
```

推荐流程：

```bash
# 终端 A: 启动 policy runner，act 或 residual 模式都可以
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path /path/to/act/pretrained_model
```

```bash
# 终端 B: 启动 RL correction 采集
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/bw_rl_corrections \
  --session-name rl_correction_001 \
  --task "act rollout with human correction"
```

RL 模式会保存 BC 字段，并额外保存：

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
timing.arm_action_dt
timing.gripper_action_dt
timing.action_act_dt
timing.action_final_dt
```

RL 字段来源和处理：

| 字段 | 来源 | 处理/语义 |
| --- | --- | --- |
| `control_source` | `/{robot_sn}/Teleop/control_source` | `0=REMOTE/人工遥控`，`1=INFERENCE/策略控制` |
| `is_intervention` | 由采集器计算 | `control_source == 0` 时为 1 |
| `has_human_action` | 由采集器计算 | 当前等于 `is_intervention` |
| `action.act` | `/Policy/debug/action_act` | ACT 基础策略动作，16 维关节位置空间 |
| `action.rl_delta` | `/Policy/debug/action_rl_delta` | 仍为 16 维；14 个手臂位置保存 joint-space residual，夹爪索引 7/15 固定为 0 |
| `action.human` | Teleop 手臂 + 夹爪动作 | 接管帧保存人工动作；非接管帧保存 0 向量 |
| `action.executed` | 由采集器根据控制源选择 | 接管帧等于 `action.human`；非接管帧等于 `/Policy/debug/action_final` |
| `action` | 由采集器写入 | RL 模式下等于 `action.executed`，表示本帧实际执行动作 |
| `action.gripper_policy_class` | `/Policy/debug/gripper_residual_class` | `(2,) int64`，左右原始策略类别：`0=KEEP_BASE, 1=FORCE_OPEN, 2=FORCE_CLOSE` |
| `reward` | 键盘标注 | 默认 0，按键时写入当前帧 |
| `done` | 键盘标注/停止逻辑 | episode 终止帧为 1；Ctrl+C 或 max frames 结束时最后一帧会标成失败终止 |
| `success` | 键盘标注 | 成功终止帧为 1，失败终止帧为 0 |
| `timing.*` | ROS header stamp 差值 | 各动作/debug topic stamp 减 state stamp，仅用于诊断；如果发布端 stamp 没填好，该值可能没有物理意义 |

键盘 reward 规则：

```text
a: 左块完成，reward += 1，episode 不结束
d: 右块完成，reward += 2，done=1，success=1，结束 episode
s: 两块都进入盒子且右块叠在左块上，reward += 3，done=1，success=1，结束 episode
g: 手动成功，reward += 1，done=1，success=1，结束 episode
j: 手动失败，reward += 0，done=1，success=0，结束 episode
```

正常完成时按 `a` 后再按 `d`，episode 总 reward 为 `3`。堆叠完成时按 `a`
后再按 `s`，episode 总 reward 为 `4`；`s` 的终止帧奖励 `3` 同时包含右块完成、
任务成功和堆叠额外奖励。

## 16 维关节顺序

所有 16 维向量字段都使用同一个顺序：

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

采集器按 `JointState.name` 做名字映射，而不是直接相信 ROS message 的数组下标。流程是：

```text
JointState.name + JointState.position
-> 关节名 alias 归一化
-> 检查重复、缺失、未知关节和 NaN/Inf
-> dict[name] = position
-> 按上面的固定顺序组装 16 维 float32
```

因此，ROS topic 里关节顺序可以不同；只要名字能映射上，保存出来的顺序都是上面的固定顺序。已知 alias 包括 `left_elbow_pitch_joint -> left_elbow_joint`、`right_elbow_pitch_joint -> right_elbow_joint`、`left_gripper -> left_gripper_joint`、`right_gripper -> right_gripper_joint` 等。

## Residual 动作语义

RL 采集器本身不发布控制动作，它只记录 policy runner 和 Teleop/机器人系统中的消息。真正发布动作的是 `lerobot_bw_policy_runner`。

policy runner 的手臂 residual 合成逻辑是：

```text
delta_norm      = residual_policy(obs)              # 归一化 residual，通常在 [-1, 1]
delta_joint     = clip(delta_norm, -1, 1) * residual_limits
action_final_raw = action.act + residual_lambda * delta_joint
action_final_arm = clamp/smoothing(action_final_raw_arm)
```

采集到的 `action.rl_delta` 是 `delta_joint`，单位和 `action.act` 兼容，都是关节位置空间的量。但最终合成时加到 ACT 上的是 `residual_lambda * action.rl_delta`，不是直接加完整 `action.rl_delta`。

如果 policy runner 开启了 smoothing 或 clamp，非接管帧的 `action.executed`/`action_final` 可能不严格等于：

```text
action.act + residual_lambda * action.rl_delta
```

这个关系只检查 14 个手臂维度。夹爪绕过连续 residual 与 smoothing，最终值只能是
`0.0/0.8`；`action.act` 的夹爪仍保留原始连续 ACT 输出。

旧纠正数据必须先非原地升级后才能和新数据合并。升级器只接受所有帧
`action.rl_delta[7] == action.rl_delta[15] == 0` 的数据，并补入 `KEEP_BASE`，不会推断旧类别：

```bash
python3 tools/upgrade_gripper_class_schema.py OLD_DATASET NEW_DATASET
```

## 落盘结构

采集完成后，一个 session 是标准 LeRobot 数据集：

```text
<dataset_root>/<session_name>/
  data/chunk-000/file-000.parquet
  videos/observation.images.env_cam/chunk-000/file-000.mp4
  videos/observation.images.left_wrist_cam/chunk-000/file-000.mp4
  videos/observation.images.right_wrist_cam/chunk-000/file-000.mp4
  meta/info.json
  meta/tasks.parquet
  meta/stats.json
  meta/episodes/chunk-000/file-000.parquet
```

`meta/info.json` 记录所有 feature 的 shape/dtype/names。LeRobot 还会自动增加这些标准列：

```text
timestamp
frame_index
episode_index
index
task_index
```

其中 `timestamp = frame_index / fps`，不是 ROS header stamp。三路图像默认以 mp4 形式保存，parquet 中主要保存非图像字段和索引。

## 采集后检查

BC 数据集：

```bash
./scripts/check_dataset.sh \
  ~/robot_datasets/bw_lerobot/bc_demo_001 \
  --mode bc \
  --save-csv
```

RL 数据集：

```bash
./scripts/check_dataset.sh \
  ~/robot_datasets/bw_rl_corrections/rl_correction_001 \
  --mode rl \
  --episode 0 \
  --save-csv
```

检查脚本会生成 overview、CSV 和图片，用来查看：

- 字段是否齐全；
- 16 维状态/动作是否有 NaN/Inf；
- 图像帧数是否和 parquet 行数匹配；
- RL 模式下 `control_source/is_intervention` 是否一致；
- 接管帧 `action.executed` 是否等于 `action.human`；
- 非接管帧 14 维手臂 `action.executed` 与 `action.act + lambda * action.rl_delta` 的差异；
- 夹爪 delta 是否为零、最终端点、类别值域与独立分类事件数；
- reward/done/success 事件是否合理。

## 后续训练如何使用这些字段

`lerobot_bw_rl` 会读取 RL 数据集构造 residual 训练数据：

```text
obs_t = [ACT 视觉特征, observation.state, action.act]
```

Residual BC 的监督目标：

```text
接管帧:
arm_target = (action.human - action.act)[arm_indices] / residual_lambda / residual_limits
gripper_target = KEEP_BASE / FORCE_OPEN / FORCE_CLOSE

非接管帧:
target = 0
```

Residual SAC 的 transition：

```text
obs_t      = [visual_t, state_t, action.act_t]
action_t   = 14-D normalized arm residual
reward_t   = reward
next_obs_t = [visual_t+1, state_t+1, action.act_t+1]
done_t     = done
```

终止帧会保留为 self-loop transition，以免丢失最终 reward。

## 常见注意事项

- BC/RL 的 `observation.state` 和 `action*` 都是原始关节位置数值按名字重排后的结果，不在采集器里归一化。
- RL 模式需要 policy runner 持续发布 debug topic；否则 `require_rl_debug_topics=true` 时采集器不会组出有效样本。
- `control_source=0` 表示人工接管，`control_source=1` 表示策略控制。这个选择也会影响底层串口节点实际采用 Teleop 还是 Policy 动作。
- 非接管帧的 `action.human` 是 0 向量，要结合 `has_human_action` 判断它是否有效人工动作。
- 当前代码采集时直接写 `reward/done/success`，旧的 annotation JSON 流程不要再作为主流程使用。
