# BW 训练全流程操作手册

> 适用：ROS 2 Humble、LeRobot 0.4.4、Python 3.10、`BW_IZN3E0FU`。
>
> RL 部分是离线流程：先采集，再合并，最后训练。不会边控制机器人边更新模型。
>
> 本手册只适用于第三代相机合同：环境 D435 `640x480@30`、左右 D405 `480x270@30`。旧数据集和旧 ACT/Residual checkpoint 不兼容，必须从 BC 示教重新采集和训练。

## 一、流程概览

```text
启动底层和相机
-> 采集 BC 示教
-> 合并 BC 数据
-> 训练 ACT
-> 运行 ACT，采集人工纠正数据
-> 合并纠正数据
-> 训练 Residual BC
-> 运行 ACT + Residual BC，采集 RL rollout
-> 合并 RL rollout
-> 训练 Offline Residual SAC/CQL
-> 部署 ACT + Residual RL
```

本文沿用当前已验证的模型名和参数。同名目录在当前机器上已存在；重新采集或训练时，请换新版本名，并同步修改后续命令里的路径。

## 二、每次采集前启动程序

先确认机器人已上电，北通手柄和三路相机已连接。

```bash
./bw_teleoperate_ws/scripts/local/start_lqy.sh
```

## 三、采集 BC 示教数据

### 1. 采集前检查

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc
```

### 2. 采集一个 episode

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode bc \
  --episode-type demo \
  --dataset-root ~/robot_datasets/pick_block_to_box/ACT_Data \
  --session-name pick_block_to_box_v1_ep001 \
  --task "pick block to box"
```

- 每采一条新数据，只修改 `ep001` 编号，不能与已有目录重名。
- 示教结束时，在采集终端按 `Ctrl+C`，程序会保存当前 episode。

### 3. 检查数据

```bash
./scripts/check_dataset.sh \
  ~/robot_datasets/pick_block_to_box/ACT_Data/pick_block_to_box_v1_ep001 \
  --mode bc \
  --episode 0 \
  --save-csv
```

## 四、合并 BC 数据并训练 ACT

### 1. 合并前试运行

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_gen3 \
  --out ~/robot_datasets/pick_block_gen3_merged \
  --repo-id local/pick_block_gen3_merged \
  --mode bc \
  --include 'pick_block_to_box_v1_ep*' \
  --dry-run
```

### 2. 正式合并

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_to_box/ACT_Data \
  --out ~/robot_datasets/pick_block_to_box/ACT_Data/pick_block_to_box_merged \
  --repo-id local/pick_block_to_box \
  --mode bc \
  --include 'pick_block_to_box_v1_ep*'
```

输出目录已存在时，脚本会拒绝覆盖。新增 episode 后优先使用新的合并目录名；只有确认旧结果可删除时，才使用 `--force`。

### 3. 训练 ACT

```bash
source ~/venvs/lerobot_ros310/bin/activate

lerobot-train \
  --dataset.repo_id=local/pick_block_to_box \
  --dataset.root=$HOME/robot_datasets/pick_block_to_box/ACT_Data/pick_block_to_box_merged \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$HOME/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box \
  --job_name=act_pick_block_to_box \
  --wandb.enable=false \
  --seed=1000 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=50000 \
  --save_freq=25000 \
  --log_freq=100 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01
```

ACT 模型路径：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model
```

## 五、运行 ACT 并采集 Residual 纠正数据

### 1. 终端 C：先 dry-run，再启动 ACT

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30 \
  --dry-run \
  --max-steps 30
```

dry-run 通过后：

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

### 2. 终端 D：采集 correction

必须在 policy runner 已运行后检查 RL 话题：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl
```

检查通过后采集：

```bash
./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/bw_rl_corrections \
  --session-name rl_correction_031 \
  --task "act rollout with human correction"
```

| 按键 | 作用 |
| --- | --- |
| 手柄 `+` | 切换人工/策略控制，以终端打印的 `control_source` 为准 |
| 手柄 `-` | 退回人工控制 |
| 采集终端 `a` | 左阶段完成，reward `+1`，不结束 |
| 采集终端 `d` | 右阶段完成，总 reward `+2`，成功结束 |
| 采集终端 `g` | 手动成功，reward `+1`，结束 |
| 采集终端 `j` | 手动失败，reward `+0`，结束 |

RL 成功 episode 使用 `d` 或 `g` 结束，失败使用 `j`。`Ctrl+C` 或 `--max-frames` 会把最后一帧标记为失败。reward/done/success 已直接写入数据集，不需要旧的 annotation JSON 流程。

## 六、合并纠正数据并训练 Residual BC

### 1. 批量检查

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

bash scripts/batch_check_rl_datasets.sh \
  ~/robot_datasets/bw_rl_corrections \
  ~/robot_datasets/bw_rl_corrections_viz
```

### 2. 合并

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/bw_rl_corrections \
  --out ~/robot_datasets/bw_rl_corrections_merged \
  --repo-id local/bw_rl_corrections_merged \
  --mode rl \
  --include 'rl_correction_*' \
  --dry-run
```

去掉 `--dry-run` 后正式合并：

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/bw_rl_corrections \
  --out ~/robot_datasets/bw_rl_corrections_merged \
  --repo-id local/bw_rl_corrections_merged \
  --mode rl \
  --include 'rl_correction_*'
```

### 3. 检查合并结果

```bash
./scripts/check_dataset.sh \
  ~/robot_datasets/bw_rl_corrections_merged \
  --mode rl \
  --all-episodes \
  --save-csv

cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl
./scripts/check_rl_rewards.sh \
  ~/robot_datasets/bw_rl_corrections_merged \
  --show-events
```

### 4. 训练 Residual BC

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl

./scripts/train_bc.sh \
  --dataset.root ~/robot_datasets/bw_rl_corrections_merged \
  --dataset.repo_id local/bw_rl_corrections_merged \
  --act-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --output_dir ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3 \
  --device cuda \
  --seed 42 \
  --steps 80000 \
  --batch_size 256 \
  --hidden_dims 256 256 \
  --intervention-ratio 0.5 \
  --intervention-loss-weight 3.0 \
  --residual-lambda 1.0 \
  --residual-limit-default 0.20 \
  --residual-limit-gripper 0.30 \
  --visual-feature-mode cache \
  --save_freq 2000 \
  --log_freq 100 \
  --num_workers 2
```

训练会自动生成 ACT 视觉特征缓存。输出模型：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt
```

## 七、用 Residual BC 采集 RL rollout

### 1. 终端 C：启动 ACT + Residual BC

先用下列命令 dry-run：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt \
  --device cuda \
  --fps 30 \
  --dry-run \
  --max-steps 30
```

dry-run 通过后，真实运行：

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt \
  --device cuda \
  --fps 30
```

### 2. 终端 D：采集 rollout

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl

./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type rollout \
  --dataset-root ~/robot_datasets/bw_residual_rl_rollouts \
  --session-name rl_rollout_001 \
  --task "act residual bc rollout"
```

按键与第五节一致。同时保留成功和失败 episode；策略即将出错时应立即切回人工控制纠正。

## 八、合并 RL rollout 并训练 Residual RL

### 1. 合并 rollout

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/bw_residual_rl_rollouts \
  --out ~/robot_datasets/bw_residual_rl_rollouts_merged \
  --repo-id local/bw_residual_rl_rollouts_merged \
  --mode rl \
  --include 'rl_rollout_*' \
  --dry-run
```

去掉 `--dry-run` 后正式合并：

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/bw_residual_rl_rollouts \
  --out ~/robot_datasets/bw_residual_rl_rollouts_merged \
  --repo-id local/bw_residual_rl_rollouts_merged \
  --mode rl \
  --include 'rl_rollout_*'
```

### 2. 检查 reward/done/success

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl

./scripts/check_rl_rewards.sh \
  ~/robot_datasets/bw_residual_rl_rollouts_merged \
  --show-events \
  --strict
```

每个 episode 最后一帧应为 `done=1`；成功轨迹应为 `success=1`，且存在非零 reward。

### 3. 训练 Offline Residual SAC/CQL

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl

./scripts/train_sac.sh \
  --dataset.root ~/robot_datasets/bw_residual_rl_rollouts_merged \
  --dataset.repo_id local/bw_residual_rl_rollouts_merged \
  --act-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --init-from-bc ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt \
  --output_dir ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_v1 \
  --device cuda \
  --seed 42 \
  --steps 30000 \
  --batch_size 256 \
  --hidden_dims 256 256 \
  --residual-lambda 1.0 \
  --residual-limit-default 0.20 \
  --residual-limit-gripper 0.30 \
  --bc-loss-weight 0.1 \
  --cql-alpha 0.1 \
  --visual-feature-mode cache \
  --save_freq 2000 \
  --log_freq 100 \
  --num_workers 2
```

输出模型：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_v1/checkpoints/last/residual_rl.pt
```

## 九、最终部署 ACT + Residual RL

先 dry-run：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_v1/checkpoints/last/residual_rl.pt \
  --device cuda \
  --fps 30 \
  --dry-run \
  --max-steps 30
```

dry-run 通过后，真实运行：

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_v1/checkpoints/last/residual_rl.pt \
  --device cuda \
  --fps 30
```

## 十、必须保持一致的内容

1. BC 数据和 RL 数据 schema 不同，不能合并在一起。
2. Residual BC、Residual RL 必须使用完全相同的 ACT checkpoint。
3. Residual BC 和 Residual RL 的 `hidden_dims`、lambda 和 limits 必须一致。
4. 本文当前参数是 `hidden_dims=256 256`、`lambda=1.0`、手臂 limit `0.20`、夹爪 limit `0.30`。
5. RL 采集必须先启动 policy runner，否则缺少 `Policy/debug/*` 话题。
