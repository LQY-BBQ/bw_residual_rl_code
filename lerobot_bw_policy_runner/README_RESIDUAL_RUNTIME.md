# lerobot_bw_policy_runner：ACT / ACT+Residual BC / ACT+Residual RL

支持三种部署模式：

```text
act
act_residual_bc
act_residual_rl
```

旧名称 `act_residual_sac` 保留为 `act_residual_rl` 的别名。

最终控制话题不变：

```text
/{robot_sn}/Policy/joint_angle_solution/smooth
/{robot_sn}/Policy/gripper_pos
```

调试话题继续用于 RL 数据采集：

```text
/{robot_sn}/Policy/debug/action_act
/{robot_sn}/Policy/debug/action_rl_delta
/{robot_sn}/Policy/debug/action_final
```

## 1. 每个控制周期都重新运行 ACT

现在三种模式都不会使用 ACT 的长动作队列。每个控制周期都会重新执行 ACT：

- 如果 ACT 配置了 temporal ensemble，继续使用 LeRobot 的 temporal ensemble；
- 如果没有 temporal ensemble，则每次重新预测 chunk，只执行该 chunk 的第一个动作。

你当前 ACT 的训练参数：

```text
chunk_size=30
n_action_steps=1
temporal_ensemble_coeff=0.01
```

正好符合“每步重新推理 + temporal ensemble”的运行方式。

## 2. Residual 模式只进行一次图像编码

在 `act_residual_bc` 和 `act_residual_rl` 中，同一次 ACT forward 同时得到：

```text
ACT动作 action_ACT
ACT三路投影视觉特征
```

实现方式是在 ACT 的 `encoder_img_feat_input_proj` 上注册精确 hook，捕获 ACT 本次前向已经计算出的三路特征图，再做空间平均池化。不会为 residual 网络再次运行 ResNet。

流程为：

```text
三路图像
  -> ACT preprocessor
  -> ACT ResNet + 1x1 Conv（只运行一次）
       -> ACT Transformer -> action_ACT
       -> pooled visual feature -> residual BC/RL

final = action_ACT + lambda × residual_limits × residual_norm
```

## 3. 只运行 ACT

```bash
cd ~/mycode/bw_residual_rl_code_package/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path ~/outputs/train/act_pick_block_0617_v2/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

## 4. ACT + Residual BC

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path ~/outputs/train/act_pick_block_0617_v2/checkpoints/last/pretrained_model \
  --residual-policy-path ~/outputs/train/residual_bc_pick_block_30_v1/checkpoints/last/residual_bc.pt \
  --residual-lambda 0.2 \
  --device cuda \
  --fps 30
```

## 5. ACT + Residual RL

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/outputs/train/act_pick_block_0617_v2/checkpoints/last/pretrained_model \
  --residual-policy-path ~/outputs/train/residual_rl_pick_block/checkpoints/last/residual_rl.pt \
  --device cuda \
  --fps 30
```

默认从 residual checkpoint 读取：

```text
residual_limits
residual_lambda
输入归一化统计量
ACT fingerprint
相机顺序
```

只有显式传入 `--residual-lambda` 时，才覆盖 checkpoint 中的 lambda。Residual limits 始终使用 checkpoint 中保存的训练值，防止训练和部署不一致。

## 6. 安全检查

Residual 模式启动时会检查：

- 当前 ACT checkpoint 与 residual checkpoint 中的 ACT 指纹完全一致；
- 三路相机字段和顺序一致；
- 视觉特征维度一致；
- 模式与 checkpoint 类型一致；
- 输入归一化维度一致；
- state/action 都是 16 维。

不一致时程序会直接停止，不会带着错误模型控制机器人。
