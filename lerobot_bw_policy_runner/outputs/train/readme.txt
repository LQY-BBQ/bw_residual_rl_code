	act_pick_block_0617使用的指令
lerobot-train \
  --dataset.repo_id=local/pick_block_0617_merged \
  --dataset.root=$HOME/robot_datasets/pick_block_0617_merged \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$HOME/outputs/train/act_pick_block_0617 \
  --job_name=act_pick_block_0617 \
  --wandb.enable=false \
  --batch_size=4 \
  --num_workers=2 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=50 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30
  运行结果： ep:83 epch:1.67 loss:0.112 grdn:11.912 lr:1.0e-05
  
  	act_pick_block_0617_V2使用的指令
lerobot-train \
  --dataset.repo_id=local/pick_block_0617_merged \
  --dataset.root=$HOME/robot_datasets/pick_block_0617_merged \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=$HOME/outputs/train/act_pick_block_0617_v2 \
  --job_name=act_pick_block_0617_V2 \
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
