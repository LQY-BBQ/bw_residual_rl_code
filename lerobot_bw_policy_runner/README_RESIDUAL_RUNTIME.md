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
/{robot_sn}/Policy/debug/action_composed
/{robot_sn}/Policy/debug/action_final
```

## 摄像头分辨率适配

当前三路实时输入为：

| 相机 | ROS 输入 | 部署端处理 |
| --- | --- | --- |
| `env_cam` | `/camera/env_d435/color/image_raw`，D435 `640x480 rgb8` | 必须与 ACT `(3,480,640)` 完全一致 |
| `left_wrist_cam` | `/camera/left_d405/color/image_raw`，D405 `480x270 rgb8` | 必须与 ACT `(3,270,480)` 完全一致 |
| `right_wrist_cam` | `/camera/right_d405/color/image_raw`，D405 `480x270 rgb8` | 必须与 ACT `(3,270,480)` 完全一致 |

部署端会读取 ROS `Image.step` 并移除行填充。实时 ROS 帧、ACT checkpoint 和 Residual checkpoint 必须同时符合第三代相机合同；部署时不做图像裁剪或缩放。旧 ACT 和旧 Residual checkpoint 会在发布控制动作前被拒绝。

启动时会用 2 秒窗口验证三路唯一帧率均不低于 `28.5 FPS`。推理循环只在三路都收到新 header 时间戳后执行，绝不重复推理旧图；每 30 步日志同时打印实际推理 Hz、错过的 30 Hz deadline 和等待相机次数，低于门槛时输出 warning。

建议启动顺序：

```bash
# 终端 A：三路相机
ros2 launch mantis_camera_ros mantis_cameras.launch.py

# 终端 B：先以 dry-run 验证图像和模型，不发布控制命令
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path /path/to/act/pretrained_model \
  --device cuda \
  --fps 30 \
  --dry-run \
  --max-steps 30
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

四个调试话题的含义：

```text
action_act       = ACT 输出
action_rl_delta  = residual_limits × residual_norm
action_composed  = action_act + lambda × action_rl_delta
action_final     = action_composed 经过 clamp 和 smoothing 后的最终下发命令
```

## 3. 只运行 ACT

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner

./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act \
  --policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --device cuda \
  --fps 30
```

## 4. ACT + Residual BC

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_bc \
  --policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --residual-policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_gen3/checkpoints/last/residual_bc.pt \
  --device cuda \
  --fps 30
```

## 5. ACT + Residual RL

```bash
./scripts/run_infer.sh \
  --robot-sn BW_IZN3E0FU \
  --mode act_residual_rl \
  --policy-path ~/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
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
- 数据集源分辨率与实时三路源分辨率一致（新 checkpoint）；
- ACT 三路目标尺寸与训练时一致（新 checkpoint）；
- 第三代相机合同版本、三路尺寸和 `none_exact_shape` 标记一致；
- state/action 都是 16 维。

不一致时程序会直接停止，不会带着错误模型控制机器人。

## 7. 实时动作可视化

可视化是独立 ROS2 进程，只订阅四个调试话题，不加载策略模型、不占用 GPU，也不会发布机器人控制命令。

第一次使用时安装可选依赖：

```bash
cd ~/mycode/bw_residual_rl_code/lerobot_bw_policy_runner
source ~/venvs/lerobot_ros310/bin/activate
python3 -m pip install -e '.[visualization]'
```

启动推理节点后，在另一个终端运行：

```bash
./scripts/run_action_visualizer.sh \
  --robot-sn BW_IZN3E0FU \
  --window-seconds 10 \
  --refresh-hz 20
```

窗口包含四个可选择关节的面板。每个面板同时显示 ACT、ACT+有效残差、最终下发命令三条曲线，以及原始残差、有效残差和后处理差值。这里的 `Final command` 是下发给控制器的命令，不是关节反馈位置。
