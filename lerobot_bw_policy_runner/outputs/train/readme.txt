act_pick_block_gen3 使用的指令
lerobot-train \
  --dataset.repo_id=local/pick_block_gen3_merged \
  --dataset.root=$HOME/robot_datasets/pick_block_gen3_merged \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$HOME/outputs/train/act_pick_block_gen3 \
  --job_name=act_pick_block_gen3 \
  --wandb.enable=false \
  --batch_size=4 \
  --num_workers=2 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=50 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30
  运行结果： ep:83 epch:1.67 loss:0.112 grdn:11.912 lr:1.0e-05

act_pick_block_gen3 使用的指令
lerobot-train \
  --dataset.repo_id=local/pick_block_gen3_merged \
  --dataset.root=$HOME/robot_datasets/pick_block_gen3_merged \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$HOME/outputs/train/act_pick_block_gen3 \
  --job_name=act_pick_block_gen3 \
  --wandb.enable=false \
  --batch_size=8 \
  --num_workers=2 \
  --steps=50000 \
  --save_freq=10000 \
  --log_freq=100 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01
    运行结果： ep:416 epch:8.33 loss:0.049 grdn:3.243 lr:1.0e-05
    
    residual_bc_pick_block_30_v1使用的指令：
bash scripts/train_bc.sh \
  --dataset.root "$DATA_ROOT" \
  --dataset.repo_id local/bw_rl_corrections_merged \
  --act-policy-path "$ACT_PATH" \
  --output_dir "$BC_OUT" \
  --device cuda \
  --seed 42 \
  --steps 80000 \
  --batch_size 128 \
  --hidden_dims 512 512 256 \
  --lr 3e-4 \
  --weight_decay 1e-5 \
  --intervention-ratio 0.5 \
  --intervention-loss-weight 3.0 \
  --residual-lambda 0.2 \
  --residual-limit-default 0.03 \
  --residual-limit-gripper 0.03 \
  --normalization-clip 5.0 \
  --visual-feature-mode cache \
  --visual-cache-dir "$CACHE_ROOT" \
  --save_freq 2000 \
  --log_freq 100 \
  --num_workers 2 \
  2>&1 | tee "$HOME/train_residual_bc_pick_block_30.log"
    
    
    
    residual_bc_pick_block_30_v2使用的指令：
source /home/lanchong/venvs/lerobot_ros310/bin/activate

cd /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_rl

python3 train_residual_bc.py \
  --dataset.root /home/lanchong/robot_datasets/bw_rl_corrections_merged \
  --act-policy-path /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/act_pick_block_gen3/checkpoints/last/pretrained_model \
  --output_dir /home/lanchong/mycode/bw_residual_rl_code/lerobot_bw_policy_runner/outputs/train/residual_bc_pick_block_30_v2 \
  --device cuda \
  --steps 80000 \
  --batch_size 256 \
  --intervention-ratio 0.5 \
  --intervention-loss-weight 3.0 \
  --residual-lambda 1.0 \
  --residual-limit-default 0.20 \
  --residual-limit-gripper 0.30 \
  --visual-feature-mode cache
  
  
	act_pick_block_to_box使用的指令：
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
  运行结果
  Training: 100%|██████████████████████▉| 49800/50000 [2:38:21<00:38,  5.25step/s]INFO 2026-07-31 02:12:17 ot_train.py:435 step:50K smpl:398K ep:326 epch:6.80 loss:0.046 grdn:3.230 lr:1.0e-05 updt_s:0.185 data_s:0.005
Training: 100%|██████████████████████▉| 49900/50000 [2:38:40<00:19,  5.25step/s]INFO 2026-07-31 02:12:36 ot_train.py:435 step:50K smpl:399K ep:327 epch:6.82 loss:0.046 grdn:3.267 lr:1.0e-05 updt_s:0.185 data_s:0.005
Training: 100%|███████████████████████| 50000/50000 [2:38:59<00:00,  5.25step/s]INFO 2026-07-31 02:12:55 ot_train.py:435 step:50K smpl:400K ep:328 epch:6.83 loss:0.046 grdn:3.342 lr:1.0e-05 updt_s:0.185 data_s:0.005
