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

### 1. 正式合并

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_to_box/ACT_Data \
  --out ~/robot_datasets/pick_block_to_box/ACT_Data/pick_block_to_box_merged \
  --repo-id local/pick_block_to_box \
  --mode bc \
  --include 'pick_block_to_box_v1_ep*'
```

输出目录已存在时，脚本会拒绝覆盖。新增 episode 后优先使用新的合并目录名；只有确认旧结果可删除时，才使用 `--force`。

### 2. 训练 ACT

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
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model
```

## 五、运行 ACT 并采集 Residual 纠正数据

### 1. 终端 C：启动 ACT

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

正式发布前还必须确认 `check_inputs.sh` 中 `Teleop/control_source` 类型正确。`Teleop/gripper_pos` 是
交接专用输入：它缺失不会阻止 ACT/residual 推理，但会阻止人工状态不完整时切回策略控制。

人工纠正后切回策略时，Runner 会清空 ACT temporal ensemble，先保持纠正姿态 6 帧，再以最多
`0.15 rad/s`（30 Hz 下每步 `0.005 rad`）平滑恢复；串口层同时等待切换后新产生的 Policy 手臂和夹爪
命令，两者都到达前继续保持人工目标。若终端持续显示 `waiting for a valid Teleop gripper command`，
不要反复切换，应检查 `/{robot_sn}/Teleop/gripper_pos` 的发布者、名称和二值命令。

### 2. 终端 D：验证执行动作

```bash
cd /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_action_visualizer.sh \
  --robot-sn BW_IZN3E0FU \
  --window-seconds 10 \
  --refresh-hz 20
```

### 3. 终端 E：采集 correction

必须在 policy runner 已运行后检查 RL 话题：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

./scripts/check_topics.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl
```

检查通过后采集：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

./scripts/collect.sh \
  --robot-sn BW_IZN3E0FU \
  --mode rl \
  --episode-type correction \
  --dataset-root ~/robot_datasets/pick_block_to_box/Res_BC_Data \
  --session-name act_correction_001 \
  --task "act rollout with human correction"
```

| 按键 | 作用 |
| --- | --- |
| 手柄 `+` | 切换人工/策略控制，以终端打印的 `control_source` 为准 |
| 手柄 `-` | 退回人工控制 |
| 采集终端 `a` | 左阶段完成，reward `+1`，不结束 |
| 采集终端 `d` | 右阶段完成，总 reward `+2`，成功结束 |
| 采集终端 `s` | 两块都进入盒子且右块叠在左块上，reward `+3`，成功结束 |
| 采集终端 `g` | 手动成功，reward `+1`，结束 |
| 采集终端 `j` | 手动失败，reward `+0`，结束 |

RL 普通成功 episode 使用 `d` 结束，堆叠成功使用 `s` 结束，通用手动成功使用 `g`，失败使用 `j`。正常按 `a` 后再按 `s` 时总 reward 为 `4`。`Ctrl+C` 或 `--max-frames` 会把最后一帧标记为失败。reward/done/success 已直接写入数据集，不需要旧的 annotation JSON 流程。

首次使用新交接逻辑时，先移除方块并清空工作区，在有人可立即按 `-` 接管的条件下，重复至少 5 次
小幅人工调整后切回策略，检查 `INITIAL_HOLD -> RESUMING -> INFERENCE` 日志和动作曲线。通过后先采集
5–10 个 pilot correction episode，刻意包含新夹爪外观和容易偏抓的方块角度；检查数据合同、切换曲线
以及 `action.executed` 后，再开始正式 Residual BC 采集。

## 六、合并纠正数据并训练 Residual BC

### 1. 检查单个纠正数据集

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

bash scripts/check_dataset.sh \
  ~/robot_datasets/pick_block_to_box/Res_BC_Data/act_correction_001 \
  --mode rl \
  --all-episodes \
  --out-dir ~/robot_datasets/pick_block_to_box/Res_BC_Data_viz/act_correction_001
```

### 2. 本批数据结论：先补采夹爪纠正

2026-08-05 对 `act_correction_001` 至 `027` 使用训练代码的同一套夹爪标签规则进行预检，独立纠正
事件数如下。这里的“事件”是一次独立的 `FORCE_OPEN` 或 `FORCE_CLOSE` 状态切换，不是处于接管状态的
帧数。

| 纠正类别 | 当前事件数 | 训练硬下限 | 最少补采 | 建议补到 25 次还需 |
| --- | ---: | ---: | ---: | ---: |
| 左夹爪打开 | 11 | 20 | 9 | 14 |
| 左夹爪关闭 | 17 | 20 | 3 | 8 |
| 右夹爪打开 | 18 | 20 | 2 | 7 |
| 右夹爪关闭 | 23 | 20 | 0 | 2 |

训练器会在生成 ACT 视觉缓存之前执行同样的预检，并禁止把 `--gripper-min-events` 调到 20 以下。
因此当前 27 组不能直接开始 Residual BC；先从 `act_correction_028` 继续补采，最低补齐前三项，推荐
四类都达到 25 个独立事件。每组补采后仍按第 1 步检查，再重新执行下面的合并流程。

`act_correction_027` 虽然策略开始后没有人工纠正，但数据合同正确且任务成功，可保留为
`KEEP_BASE`/零 residual 样本；训练 batch 固定按 50% 接管、50% 非接管采样，它不会稀释接管样本。

### 3. 合并前 dry-run

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_data_collector

bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_to_box/Res_BC_Data \
  --out ~/robot_datasets/pick_block_to_box/Res_BC_Data_merged \
  --repo-id local/pick_block_to_box_res_bc_merged \
  --mode rl \
  --include 'act_correction_*' \
  --dry-run
```

当前 27 组的 dry-run 应识别为 `27 episodes, 47114 frames`；补采后 episode 和 frame 数会相应增加。

### 4. 正式合并

dry-run 通过后，使用同一组参数去掉 `--dry-run`：

```bash
bash scripts/merge_lerobot_datasets.sh \
  --src ~/robot_datasets/pick_block_to_box/Res_BC_Data \
  --out ~/robot_datasets/pick_block_to_box/Res_BC_Data_merged \
  --repo-id local/pick_block_to_box_res_bc_merged \
  --mode rl \
  --include 'act_correction_*'
```

首次合并不要加 `--force`。输出目录已存在时，优先换一个新目录名；只有确认旧合并结果可以删除时，
才使用 `--force` 重建。

### 5. 检查合并结果

```bash
jq '{total_episodes, total_frames, fps, robot_type}' \
  ~/robot_datasets/pick_block_to_box/Res_BC_Data_merged/meta/info.json

cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl
./scripts/check_rl_rewards.sh \
  ~/robot_datasets/pick_block_to_box/Res_BC_Data_merged \
  --show-events \
  --strict
```

本批各源数据集已经完成完整 RL schema、动作关系和视频检查；合并脚本还会再次校验 schema、帧数和
输出元数据。上面的 strict reward 检查必须显示 `PASS`。

### 6. 训练 Residual BC

补齐夹爪事件并重新合并后再运行下面的命令。训练器会先打印四类夹爪事件数，达不到每类 20 次时立即
退出，不会开始构建视觉缓存。

本批人工纠正幅度不适合通用默认值 `lambda=0.2, limit=0.03`：其有效范围只有 `0.006 rad`，会让
47.76% 的纠正分量被裁剪。这里使用 `lambda=1.0, limit=0.20`，有效范围为 `0.20 rad`，本批仅约
0.474% 的纠正分量被裁剪。两项参数会写入 checkpoint，后续 Residual RL 和部署必须保持一致。

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl

./scripts/train_bc.sh \
  --dataset.root ~/robot_datasets/pick_block_to_box/Res_BC_Data_merged \
  --dataset.repo_id local/pick_block_to_box_res_bc_merged \
  --act-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --output_dir ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1 \
  --device cuda \
  --seed 42 \
  --steps 20000 \
  --batch_size 256 \
  --hidden_dims 256 256 \
  --intervention-ratio 0.5 \
  --intervention-loss-weight 3.0 \
  --residual-lambda 1.0 \
  --residual-limit-default 0.20 \
  --gripper-loss-weight 1.0 \
  --gripper-min-events 20 \
  --validation-ratio 0.2 \
  --gripper-hysteresis \
  --gripper-open-threshold 0.50 \
  --gripper-single-threshold 0.45 \
  --gripper-close-threshold 0.40 \
  --gripper-act-confirm-frames 3 \
  --visual-feature-mode cache \
  --visual-cache-batch-size 16 \
  --visual-cache-dtype float16 \
  --visual-cache-use-amp \
  --video-backend torchcodec \
  --save_freq 2000 \
  --log_freq 100 \
  --num_workers 2
```

训练会自动生成与固定 ACT `050000` checkpoint 指纹绑定的视觉特征缓存。20,000 steps 已会对这批数据
进行约百余轮等效采样；每 2,000 steps 查看 `validation_metrics.csv`，不要直接沿用旧命令的 80,000
steps。输出模型：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1/checkpoints/last/residual_bc.pt
```

## 七、用 Residual BC 采集 RL rollout

### 1. 终端 C：启动 ACT + Residual BC

先用下列命令 dry-run：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1/checkpoints/last/residual_bc.pt \
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
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1/checkpoints/last/residual_bc.pt \
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
  --act-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --init-from-bc ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1/checkpoints/last/residual_bc.pt \
  --output_dir ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_to_box_v1 \
  --device cuda \
  --seed 42 \
  --steps 30000 \
  --batch_size 256 \
  --hidden_dims 256 256 \
  --residual-lambda 1.0 \
  --residual-limit-default 0.20 \
  --bc-loss-weight 0.1 \
  --cql-alpha 0.1 \
  --visual-feature-mode cache \
  --save_freq 2000 \
  --log_freq 100 \
  --num_workers 2
```

输出模型：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_to_box_v1/checkpoints/last/residual_rl.pt
```

## 九、最终部署 ACT + Residual RL

先 dry-run：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_to_box_v1/checkpoints/last/residual_rl.pt \
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
  --policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_to_box/checkpoints/050000/pretrained_model \
  --residual-policy-path ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_rl_pick_block_to_box_v1/checkpoints/last/residual_rl.pt \
  --device cuda \
  --fps 30
```

## 十、必须保持一致的内容

1. BC 数据和 RL 数据 schema 不同，不能合并在一起。
2. Residual BC、Residual RL 必须使用完全相同的 ACT checkpoint。
3. Residual BC 和 Residual RL 的 `hidden_dims`、lambda 和 limits 必须一致。
4. 本文当前参数是 `hidden_dims=256 256`、`lambda=1.0`、手臂 limit `0.20`；夹爪使用离散三分类，
   不使用连续 residual limit。
5. RL 采集必须先启动 policy runner，否则缺少 `Policy/debug/*` 话题。
