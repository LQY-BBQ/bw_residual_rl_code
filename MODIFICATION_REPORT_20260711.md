# BW 视觉残差 BC / RL 代码修改说明

修改日期：2026-07-11

## 一、修改范围

本次只修改：

- `lerobot_bw_rl`
- `lerobot_bw_policy_runner`

`lerobot_bw_data_collector` 保持原样。

目标运行环境：LeRobot 0.4.4、Python 3.10、三路相机、16 维状态、16 维动作。

## 二、最终结构

Residual BC 和 Residual RL 的统一输入：

```text
[ACT 三路视觉特征, observation.state, action.act]
```

三路视觉特征严格复用冻结 ACT 的：

```text
ACT preprocessor
-> ACT ResNet layer4
-> ACT encoder_img_feat_input_proj
-> 每路空间平均池化
-> env / left wrist / right wrist 拼接
```

ACT 全部参数冻结，不使用纠正数据更新 ACT。

## 三、训练程序

新增：

- `lerobot_bw_rl/build_act_visual_cache.py`
- `lerobot_bw_rl/visual_cache.py`
- `lerobot_bw_rl/train_residual_bc.py`
- `lerobot_bw_rl/policies/act_shared_encoder.py`
- `lerobot_bw_rl/policies/residual_bc_policy.py`
- `lerobot_bw_rl/scripts/build_visual_cache.sh`
- `lerobot_bw_rl/scripts/train_bc.sh`

重写/扩展：

- `lerobot_bw_rl/train_residual_sac.py`
- `lerobot_bw_rl/bw_datasets/residual_transition_dataset.py`
- `lerobot_bw_rl/scripts/train_sac.sh`

Residual BC：

- 人工接管帧目标为 `(action.human-action.act)/(lambda*limits)`；
- 非接管帧目标为 0；
- batch 默认 50% 接管帧、50% 非接管帧；
- Smooth L1；
- 确定性 MLP + tanh。

Residual RL：

- 保留原离线 SAC；
- 支持 BC regularization 和 CQL；
- 可通过 `--init-from-bc` 从 Residual BC 初始化 actor；
- 终止帧不会被丢弃，最终 reward/done 会进入训练。

## 四、推理程序

支持：

```text
act
act_residual_bc
act_residual_rl
```

兼容旧别名：`act_residual_sac`。

三种模式都每个控制周期重新执行 ACT。你的 ACT 配置：

```text
chunk_size=30
n_action_steps=1
temporal_ensemble_coeff=0.01
```

会保持 LeRobot temporal ensemble，并且每一步执行一次新的 ACT forward。

Residual 模式通过 `encoder_img_feat_input_proj` 的精确 forward hook，从同一次 ACT forward 中取得三路视觉特征。因此：

```text
图像预处理 + ResNet + 1x1 Conv：每个控制周期只运行一次
```

不会为 Residual BC/RL 再执行一次视觉编码器。

## 五、兼容性保护

Residual checkpoint 保存并在部署时检查：

- ACT checkpoint SHA256 指纹；
- 三路相机字段及顺序；
- visual feature dimension；
- state/action dimension；
- observation normalization；
- residual limits 和 residual lambda；
- residual checkpoint 类型。

使用错误 ACT 或错误 residual checkpoint 时，程序会在发布控制动作前停止。

## 六、已完成测试

已通过：

1. 两个工程全部 Python 文件 `compileall`；
2. 合成数据集的 Residual BC/transition 构造；
3. 终止帧 reward/done 保留测试；
4. BC checkpoint runtime 加载和推理；
5. BC actor 参数初始化 SAC actor；
6. 单次 ACT forward 共享视觉特征测试，确认三路相机只各调用一次 backbone；
7. 推理模式别名和配置解析测试。

当前容器没有真实 ACT checkpoint、真实 LeRobot 数据集视频、ROS2 机器人运行环境，因此没有完成真实 GPU 视频缓存、真实机器人发布和闭环执行测试。部署前应先使用 `--dry-run` 检查。

## 七、文档入口

- 训练：`lerobot_bw_rl/README.md`
- 推理：`lerobot_bw_policy_runner/README_RESIDUAL_RUNTIME.md`
