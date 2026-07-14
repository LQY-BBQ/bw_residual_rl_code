# 本地代码包说明：不会修改 GitHub

本 zip 是本地生成的工程副本，不会对 GitHub 上的仓库做任何 commit、push、branch 或 PR。

包含三个目录：

```text
lerobot_bw_data_collector/   # 增加 --mode rl，仍保留原 BC 采集模式
lerobot_bw_policy_runner/    # 增加 act_residual_sac 推理模式和 debug 话题
lerobot_bw_rl/               # 新建 residual SAC 训练工程
```

## 推荐使用顺序

### 1. 部署 ACT，采集 RL correction 数据

终端 A：运行 policy_runner。先只跑 ACT 也可以，因为 debug 话题会发布 `action_act`、零 residual、`action_final`。

```bash
cd ~/mycode/lerobot_bw_policy_runner
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path ~/outputs/train/act_pick_block/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

终端 B：采集 RL 数据。机器人出错时用手柄切换人工接管，`control_source=0` 的帧会被记录为 intervention。

```bash
cd ~/mycode/lerobot_bw_data_collector
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/bw_rl_corrections \
  --task "act rollout with human correction"
```

### 2. 标注阶段 reward

```bash
cd ~/mycode/lerobot_bw_rl
python annotate_rewards.py \
  --dataset.root ~/robot_datasets/bw_rl_corrections/session_xxx \
  --episode-index 0 \
  --success true \
  --left-block-done-frame 320 \
  --right-block-done-frame 680
```

### 3. 训练 residual SAC

```bash
cd ~/mycode/lerobot_bw_rl
./scripts/train_sac.sh \
  --dataset.root ~/robot_datasets/bw_rl_corrections/session_xxx \
  --act-policy-path ~/outputs/train/act_pick_block/checkpoints/last/pretrained_model \
  --output_dir ~/outputs/train/residual_sac_pick_block \
  --device cuda \
  --steps 20000 \
  --batch_size 256
```

### 4. 部署 ACT + residual SAC

```bash
cd ~/mycode/lerobot_bw_policy_runner
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_sac \
  --policy-path ~/outputs/train/act_pick_block/checkpoints/last/pretrained_model \
  --residual-policy-path ~/outputs/train/residual_sac_pick_block/checkpoints/last/residual_sac.pt \
  --residual-lambda 0.2 \
  --device cuda \
  --fps 30
```

## 注意

- 最终发布给机器人执行的话题没有改变，仍是 `Policy/joint_angle_solution/smooth` 和 `Policy/gripper_pos`。
- 新增的 `Policy/debug/*` 话题只用于数据采集，不应该被下位机直接执行。
- 第一版 SAC 不自动使用旧纯示教数据；SAC transition 需要通过 `--mode rl` 重新采集。
- 第一版默认 residual 输入为 `[observation.state, action_ACT]`。ACT 视觉特征提取接口已经预留，但建议等主流程跑通后再打开。
