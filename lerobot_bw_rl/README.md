# lerobot_bw_rl：ACT 视觉特征共享的 Residual BC 与 Residual RL

本工程针对以下固定环境实现：

```text
LeRobot 0.4.4
Python 3.10
三路相机：env_cam、left_wrist_cam、right_wrist_cam
状态与数据集动作维度：16
Residual BC：14 维连续手臂 + 左右夹爪各 3 类
Residual RL actor/critic 动作维度：14（夹爪分类网络从 BC 冻结携带）
```

ACT 基础策略全部冻结。Residual BC 与 Residual RL 的输入统一为：

```text
[ACT三路视觉特征, observation.state, action.act]
```

第三代相机合同为：

| 相机 | 设备 | 数据集与 ACT 输入 |
| --- | --- | --- |
| `env_cam` | 环境 D435 `152222071548` | HWC `(480, 640, 3)` / CHW `(3, 480, 640)` |
| `left_wrist_cam` | 左 D405 `335122270917` | HWC `(270, 480, 3)` / CHW `(3, 270, 480)` |
| `right_wrist_cam` | 右 D405 `260322279568` | HWC `(270, 480, 3)` / CHW `(3, 270, 480)` |

训练程序会从数据集的 `meta/info.json` 读取三路尺寸和 FPS，并从 ACT checkpoint 读取三路 CHW 尺寸。两边必须严格符合上表且 FPS 必须为 `30`；特征提取前不做裁剪或缩放。旧相机数据集、旧 ACT checkpoint 和旧视觉缓存会被拒绝，第三代流程需要重新采集并训练 ACT。

视觉特征的计算严格复用 ACT 自身的处理路径：

```text
LeRobot ACT preprocessor
-> ACT ResNet backbone 的 layer4
-> ACT encoder_img_feat_input_proj（1×1 Conv）
-> 每路相机空间平均池化
-> 按 env / left wrist / right wrist 拼接
```

如果 ACT 的 `dim_model=512`，则输入维度是：

```text
3 × 512 + 16 + 16 = 1568
```

## 1. 为什么先生成视觉缓存

训练会反复读取同一批视频。如果每个优化 step 都重新解码三路视频并运行三次 ResNet，训练会非常慢。默认流程先把每一帧的冻结 ACT 视觉特征计算一次并保存：

```text
数据集/.bw_act_visual_cache/<ACT指纹>/features.npy
数据集/.bw_act_visual_cache/<ACT指纹>/metadata.json
```

缓存与以下内容绑定：

- ACT `config.json`；
- ACT `model.safetensors`；
- ACT preprocessor；
- 三路相机顺序；
- 数据集帧数。
- 数据集 FPS（当前必须为 30）；
- 数据集三路源图像尺寸；
- ACT 三路输入尺寸；
- 第三代相机合同版本与 `none_exact_shape` 图像变换标记。

更换 ACT 模型、数据集 FPS 或图像尺寸后，不匹配的缓存会被拒绝，必须用 `--rebuild-visual-cache` 重新生成。训练 checkpoint 也会保存 `dataset_fps`，部署端会拒绝不是用 30 FPS 数据训练出的新 checkpoint。

### 手动生成缓存

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_rl

./scripts/build_visual_cache.sh \
  --dataset.root ~/robot_datasets/bw_rl_corrections/rl_correction_001 \
  --dataset.repo_id local/rl_correction_001 \
  --act-policy-path ~/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --device cuda \
  --batch-size 16 \
  --dtype float16
```

训练脚本默认也会自动检查并生成缓存，因此这一步可以省略。

`--visual-feature-mode online` 表示不复用数据集目录中的已有缓存，而是在本次训练启动时重新提取一份临时特征表。它主要用于检查特征提取逻辑，不建议用于日常反复训练。

## 2. Residual BC 训练

Residual BC 的手臂目标为：

```text
人工接管帧：
residual_target = (action.human - action.act)
                  / residual_lambda
                  / residual_limits

非接管帧：
residual_target = 0
```

输出统一裁剪到 `[-1, 1]`。每个 batch 默认严格包含：

```text
50% 人工接管帧
50% 非接管帧
```

接管帧默认 loss 权重为 3，手臂损失为 Smooth L1。夹爪标签为
`KEEP_BASE/FORCE_OPEN/FORCE_CLOSE`，左右分别使用训练集频次平方根倒数加权交叉熵。
训练/验证按 episode 做 80/20 划分，四种 FORCE 侧别/方向必须同时出现在两边。
`--gripper-min-events` 默认且推荐为每侧每种 FORCE 至少 20 个独立事件。可以显式降低到任意正整数；
训练器仍会按传入值做真实预检，低于该值时在生成视觉缓存前停止。低于推荐值 20 时会打印警告，并把
本次门槛和实际事件数一起写入 checkpoint，便于区分 pilot 模型和正式模型。

当前 27 组 `Res_BC_Data` 的左开/左关、右开/右关事件数为 `11/17`、`18/23`，四类都存在。可以使用
`--gripper-min-events 10` 直接进行 pilot 训练；20 仍是推荐覆盖量，正式部署前建议四类均达到 25 次。

```bash
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
  --gripper-min-events 10 \
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

输出：

```text
~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_to_box_v1/
  checkpoints/last/residual_bc.pt
  checkpoints/last/config.json
  train_metrics.csv
  validation_metrics.csv
```

## 3. Residual RL 训练

当前实现保留原工程的 offline SAC，并支持可选 BC regularization 与 CQL penalty。输入 transition 为：

```text
obs_t      = [visual_t, state_t, action_ACT_t]
a_t        = 14-D normalized arm residual
reward_t   = 数据集 reward
obs_next   = [visual_t+1, state_t+1, action_ACT_t+1]
done_t     = 数据集 done

终止帧同样会保留为 transition。因为当前 BW 标注方式会把最终奖励和 `done=1` 写在 episode 最后一帧，终止帧使用 `obs_next=obs_t` 的自环表示；SAC 目标中的 `(1-done)` 会屏蔽该 next-Q，因此不会引入错误的跨 episode 状态，同时不会丢失最终奖励。
```

推荐从 Residual BC 初始化：

```bash
./scripts/train_sac.sh \
  --dataset.root ~/robot_datasets/bw_rl_corrections/rl_correction_001 \
  --dataset.repo_id local/rl_correction_001 \
  --act-policy-path ~/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --init-from-bc ~/outputs/train/residual_bc_pick_block/checkpoints/last/residual_bc.pt \
  --output_dir ~/outputs/train/residual_rl_pick_block \
  --device cuda \
  --steps 30000 \
  --batch_size 256 \
  --hidden_dims 256 256 \
  --residual-lambda 0.2 \
  --residual-limit-default 0.03 \
  --bc-loss-weight 0.1 \
  --cql-alpha 0.1 \
  --save_freq 2000 \
  --log_freq 100
```

从 BC 初始化时会加载：

```text
BC trunk -> SAC trunk
BC arm_mu -> SAC mu
完整 BC actor -> 独立冻结的夹爪分类网络
```

SAC 的 `log_std` 单独初始化。代码会检查 ACT 指纹、视觉维度、归一化参数、网络尺寸和残差定义，任何不一致都会停止训练。

输出：

```text
~/outputs/train/residual_rl_pick_block/
  checkpoints/last/residual_rl.pt
  checkpoints/last/config.json
  train_metrics.csv
```

## 4. 归一化

训练会计算并保存整个 residual 输入的均值和标准差：

```text
ACT视觉特征 + 16维状态 + 16维ACT动作
```

推理器读取同一组统计量进行归一化，避免训练与部署输入分布不一致。

## 5. Checkpoint 绑定关系

Residual checkpoint 会保存：

```text
policy_type
obs_dim
visual_feature_dim
hidden_dims
observation_stats
residual_limits
residual_lambda
ACT fingerprint
三路相机顺序
ACT视觉特征定义
数据集三路源图像尺寸
ACT三路目标图像尺寸
第三代相机合同版本和 `none_exact_shape` 图像变换标记
format_version=4、dataset_action_dim=16、arm action_dim=14
夹爪分类名称、迟滞标签配置，以及 RL checkpoint 中的冻结 BC 网络
```

部署时必须使用训练该 residual 策略时完全相同的 ACT checkpoint。format v4 不兼容旧的
连续夹爪 residual checkpoint，旧 checkpoint 必须重新训练。

## 6. 当前算法范围

本版本实现：

```text
Residual BC
Offline Residual SAC
可选 BC regularization
可选 CQL penalty
```

`--demo-sample-ratio` 仍是后续 RLPD 双 buffer 的保留参数，本版本没有实现完整 RLPD actor/learner/buffer 系统。
