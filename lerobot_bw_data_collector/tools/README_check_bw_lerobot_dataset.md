# check_bw_lerobot_dataset.py 使用说明

这个脚本用于检查 `lerobot_bw_data_collector` 采集得到的 LeRobot 数据集。它同时支持两类数据：

1. **BC / 模仿学习数据**：检查 `observation.state` 和 `action`，生成和旧版数据检查程序类似的关节曲线、误差统计、相关性检查、数值范围对比和相机抽帧图。
2. **RL / residual 强化学习数据**：额外检查 `action.gripper_policy_class`、16 维存储契约、夹爪零 residual 和 `0.0/0.8` 最终端点。

当前版本的 RL reward 检查**不再依赖** `annotations/episode_000000.json`。检查程序直接读取 parquet 数据里的 `reward`、`done`、`success` 字段。

## 环境准备

推荐在你的 LeRobot 环境中运行：

```bash
source ~/venvs/lerobot_ros310/bin/activate
```

需要依赖：

```bash
pip install pandas pyarrow matplotlib opencv-python
```

如果只想看关节曲线，不想生成视频抽帧图，可以加 `--no-video-sheet`，此时可以不安装 OpenCV。

## 基本用法

进入数据采集工程：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector
```

默认检查全部 episode：

```bash
bash scripts/check_dataset.sh ~/robot_datasets/bw_lerobot/session_xxx
```

只检查某一个 episode：

```bash
bash scripts/check_dataset.sh ~/robot_datasets/bw_lerobot/session_xxx --episode 0
```

显式检查全部 episode：

```bash
bash scripts/check_dataset.sh ~/robot_datasets/bw_lerobot/session_xxx --all-episodes
```

指定输出目录：

```bash
bash scripts/check_dataset.sh ~/robot_datasets/bw_rl_corrections/rl_correction_001 \
  --all-episodes \
  --out-dir ~/robot_datasets/bw_rl_corrections_viz/rl_correction_001 \
  --mode rl
```

## BC 数据检查

```bash
bash scripts/check_dataset.sh DATASET_ROOT --mode bc --episode 0 --save-csv
```

主要输出：

```text
check_report/
├── summary.csv
├── warnings.csv
├── report.html
└── episode_000/
    ├── overview.txt
    ├── per_joint_stats.csv
    ├── per_joint_action_vs_state/
    ├── all_joints_action_vs_state_grid.png
    ├── selected_action_vs_state.png
    ├── gripper_action_vs_state.png
    ├── rmse_by_joint.png
    ├── max_abs_error_by_joint.png
    ├── correlation_sign_check.png
    ├── action_state_range_comparison.png
    └── video_contact_sheet.png
```

## RL 数据检查

```bash
bash scripts/check_dataset.sh DATASET_ROOT --mode rl --all-episodes --save-csv
```

如果数据集中没有记录 residual lambda，脚本默认使用 `0.2`，也可以手动指定：

```bash
bash scripts/check_dataset.sh DATASET_ROOT --mode rl --episode 0 --residual-lambda 0.2
```

``` ys参考
bash scripts/check_dataset.sh ~/robot_datasets/bw_rl_corrections/rl_correction_002 \
  --all-episodes \
  --out-dir ~/robot_datasets/bw_rl_corrections/rl_correction_002_check \
  --mode rl
```

RL 模式会额外输出：

```text
episode_000/
├── rl_summary.txt
├── rl_episode_stats.csv
├── field_completeness.csv
├── transition_validity.csv
├── reward_events.csv
├── control_source_timeline.png
├── intervention_timeline.png
├── reward_done_success_timeline.png
├── action_source_overview.png
├── residual_delta_norm.png
├── residual_target_norm.png
├── executed_reconstruction_error.png
├── act_vs_executed_grid.png
├── act_vs_human_grid.png
├── human_correction_minus_act_grid.png
├── camera_contact_sheet_uniform.png
├── camera_contact_sheet_intervention_events.png
└── camera_contact_sheet_reward_done_events.png
```

## RL 检查逻辑

`bw_residual_rl_code` 中保存的 `action.rl_delta` 是关节空间 residual delta，不是归一化 `[-1, 1]` 动作。

脚本会检查：

```text
control_source == 0  -> is_intervention 应为 1
action.executed      -> 接管时应等于 action.human
action               -> 应等于 action.executed
reward/done/success  -> 直接从 parquet 字段读取并检查键盘标注规则
```

非接管时，14 个手臂维度理论上：

```text
action.executed ≈ action.act + residual_lambda * action.rl_delta
```

夹爪不使用这个连续公式：`action.rl_delta[7/15]` 必须为零，最终夹爪必须为 `0.0/0.8`，类别字段只能取 `{0,1,2}`。脚本还会统计左右 FORCE 类别的独立连续事件。手臂可能启用了 smoothing / clamp，因此其重建误差不一定代表数据错误。

## 键盘 reward 标注规则

当前 RL 采集程序使用键盘直接把 reward 写入 parquet：

```text
a：左手物块放置完成，当前帧 reward += 1，done=0，success=0
d：右手物块放置完成，当前帧 reward += 2，done=1，success=1，并自动停止保存
s：两块都进入盒子且右块叠在左块上，当前帧 reward += 3，done=1，success=1，并自动停止保存
g：手动标记成功，当前帧 reward += 1，done=1，success=1，并停止保存
j：手动标记失败，当前帧 reward += 0，done=1，success=0，并停止保存
```

因此，一个正常的成功 episode 常见两种情况：

```text
按 a 后再按 d：总 reward = 3，最后一帧 reward = 2，done=1，success=1
按 a 后再按 s：总 reward = 4，最后一帧 reward = 3，done=1，success=1
只按 g 成功结束：总 reward = 1，最后一帧 reward = 1，done=1，success=1
```

一个失败 episode 的常见情况：

```text
直接按 j：总 reward = 0，最后一帧 reward = 0，done=1，success=0
按 a 后失败再按 j：总 reward = 1，最后一帧 reward = 0，done=1，success=0
```

## reward_events.csv

RL 检查会生成：

```text
episode_000/reward_events.csv
```

它记录所有 reward/done/success 事件帧，例如：

```text
frame,reward,done,success,event_type_guess
120,1.0,0.0,0.0,left_stage_done_key_a
240,2.0,1.0,1.0,right_stage_done_success_key_d
```

`rl_episode_stats.csv` 和总的 `summary.csv` 里也会增加：

```text
reward_sum_parquet
reward_event_count
reward_nonzero_frames
last_reward
last_done
last_success
terminal_event_type
keyboard_left_stage_events
keyboard_right_success_events
keyboard_manual_success_events
keyboard_failure_events
```

这样你不需要打开图像，也能快速判断一条 RL 数据是否按键标注正常。
