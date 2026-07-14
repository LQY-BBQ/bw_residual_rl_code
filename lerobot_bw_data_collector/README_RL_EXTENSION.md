# lerobot_bw_data_collector：BC + Residual RL 采集扩展

这个版本不修改 GitHub 仓库，只是在本地工程副本中增加 `--mode rl`。

## BC 模式，保持原来行为

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc \
  --task "pick block demo"
```

采集字段仍是：

```text
observation.state
observation.images.env_cam
observation.images.left_wrist_cam
observation.images.right_wrist_cam
action
```

## RL 模式

先启动 `lerobot_bw_policy_runner`，并确保它发布 debug 话题：

```text
/{robot_sn}/Policy/debug/action_act
/{robot_sn}/Policy/debug/action_rl_delta
/{robot_sn}/Policy/debug/action_final
```

再采集：

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --task "act rollout with human correction"
```

```参考以下指令 ys```
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/bw_rl_corrections \
  --session-name rl_correction_002 \
  --task "act rollout with human correction"

RL 模式额外保存：

```text
control_source
is_intervention
has_human_action
action.act
action.rl_delta
action.human
action.executed
reward
done
success
timing.*
```

说明：

- `control_source == 0`：人工遥控生效，`is_intervention=1`，`action.executed=action.human`。
- `control_source == 1`：策略推理生效，`is_intervention=0`，`action.executed=action_final`。
- 非人工接管帧的 `action.human` 保存为 0 向量，并用 `has_human_action=0` 区分。
- 录制结束的最后一帧会被标记为 `done=1`。
- `reward/success` 先写默认值，后续通过 `lerobot_bw_rl/annotate_rewards.py` 和 annotation JSON 计算训练 reward。

## 标注文件

RL 数据集保存后，会生成：

```text
annotations/episode_000000.json
```

格式：

```json
{
  "episode_index": 0,
  "episode_type": "correction",
  "success": false,
  "left_block_done_frame": null,
  "right_block_done_frame": null,
  "failure_type": "",
  "notes": "Fill this file after recording. Frame numbers are zero-based within this episode."
}
```

你需要把 `left_block_done_frame`、`right_block_done_frame` 改成对应阶段成功的帧号。
