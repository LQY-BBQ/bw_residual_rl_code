> 仅适用于第三代相机合同：环境 D435 `640x480@30`、左右 D405 `480x270@30`。不要使用旧相机数据训练出的 checkpoint。

## 1. 启动程序

打开新终端输入指令：
./bw_teleoperate_ws/scripts/local/start_lqy.sh

## 2. 开启推理

1. 只运行 ACT

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

2. ACT + Residual BC

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt \
  --device cuda \
  --fps 30
```

3. ACT + Residual RL

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path ~/outputs/train/residual_rl_pick_block/checkpoints/last/residual_rl.pt \
  --device cuda \
  --fps 30
```

## 3. 启动采集程序

1. BC 模式使用

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc \
  --dataset-root ~/robot_datasets/bw_lerobot \
  --session-name bc_demo_001 \
  --task "pick block demo"
```

2. RL 模式使用

RL 采集需要先启动 `lerobot_bw_policy_runner`

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/bw_rl_corrections \
  --session-name rl_correction_001 \
  --task "act rollout with human correction"
```

3. 采集后检查

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

4. 数据集合并


```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_gen3 \
  --out ~/robot_datasets/pick_block_gen3_merged \
  --repo-id local/pick_block_gen3_merged \
  --mode bc \
  --include 'pick_block_to_box_v1_ep*'
```

## 4. 模型训练
