### ViT-B grafting
Pretrain (train with 8 nodes with 64 V100s in total), [Pretrained Model](https://www.dropbox.com/sh/e9czo0xtivdqvff/AAACFSQPRthsZo_nNAWD0ot4a?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain=pretrain/mae_pretrain_vit_base.pth # path to the mae pre-training
pretrain_name=mae_pretrain_vit_base

lr_layer_wise="1.5e-5,1.5e-5,1.5e-4"

RANK="CurrentMachineRank"
MASTER="YourMasterIP"

bash cmds/shell_scripts/pretrain.sh --arch vit_base --save_dir . \
--batch_size 4096 -g 8 --epochs 300 --dataset imagenet \
--graft_pretrained ${pretrain} --graft_pretrained_name ${pretrain_name} \
--lr_layer_wise ${lr_layer_wise} --lr 1 \
--resume local --skip_tune True \
--data ${DATA_PATH} \
--NODE_NUM 8 --RANK ${RANK} --MASTER ${MASTER}
```

Linear eval (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/b7g9c3kbp5y4j7f/AAB_Tp1fC1ezGMXbth5u0moma?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_base_imagenet_lr1B4096E300_graftFmae_pretrain_vit_base_lrLayerW1.5e-5,1.5e-5,1.5e-4

bash cmds/shell_scripts/tunePretrained_sweep.sh --pretrain_name ${pretrain_name} \
--dataset imagenet --batch_size 4096 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
--save_dir ./checkpoints_tune --arch vit_base -g 8 -p 4833 --data ${DATA_PATH} --resume local
```

Finetune (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/go0635x6bx97z2y/AAANtU0mSPpL2NAcBqiHuoAxa?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_base_imagenet_lr1B4096E300_graftFmae_pretrain_vit_base_lrLayerW1.5e-5,1.5e-5,1.5e-4

bash cmds/shell_scripts/tunePretrained_mae.sh --pretrain_name ${pretrain_name} --dataset imagenet \
--batch_size 1024 --lr 5e-4 --layer_decay 0.6 --epochs 100 \
 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
--save_dir ./checkpoints_tune --arch vit_base -g 8 -p 4833 \
--data ${DATA_PATH} --resume local
```


1% Few shot (train with 1 V100), [Checkpoint](https://www.dropbox.com/sh/crb7058rxo69i2x/AACLa8TKeA-t7-6qTQ93MF9na?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_base_imagenet_lr1B4096E300_graftFmae_pretrain_vit_base_lrLayerW1.5e-5,1.5e-5,1.5e-4

save_dir="."

bash cmds/shell_scripts/tunePretrained_sweep.sh --pretrain_name ${pretrain_name} --dataset imagenet --batch_size 4096 \
--pretrain ${save_dir}/checkpoints/${pretrain_name}/checkpoint_final.pth.tar  \
--save_dir ${save_dir}/checkpoints_tune_repro --arch vit_base -g 1 -p 5863 --data ${DATA_PATH} \
--customSplit imagenet_1percent --customSplitName 1perc --batch_size 256 --logstic_reg True
```

10% Few shot (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/3rl3w1nqs88w8lx/AACAAyU8bosnllRealCoc3qJa?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_base_imagenet_lr1B4096E300_graftFmae_pretrain_vit_base_lrLayerW1.5e-5,1.5e-5,1.5e-4

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7  bash cmds/shell_scripts/tunePretrained_mae.sh --pretrain_name ${pretrain_name} --dataset imagenet \
--batch_size 1022 --lr 3e-5 --layer_decay 0.75 --epochs 200 --tuneFromFirstFC True \
 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
 --customSplit "imagenet_10percent" --customSplitName "10perc" --test_interval 10 \
--save_dir ./checkpoints_tune_repro --arch vit_base -g 7 -p 4833 \
--data ${DATA_PATH} --resume local
```

### ViT-L grafting 
Pretrain (train with 10 nodes with 80 V100s in total), [Pretrained Model](https://www.dropbox.com/sh/fk92wphgu8fq772/AADIffCTMlRyafva_Ungr_n4a?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain=pretrain/mae_pretrain_vit_large.pth # path to the mae pre-training
pretrain_name=mae_pretrain_vit_large

l1_dist_w=1e-5
lr_layer_wise="1.5e-5,1.5e-5,1.5e-4"

RANK="CurrentMachineRank"
MASTER="YourMasterIP"

bash cmds/shell_scripts/pretrain.sh --arch vit_large --save_dir . \
--batch_size 2080 -g 8 --epochs 300 --dataset imagenet \
--graft_pretrained ${pretrain} --graft_pretrained_name ${pretrain_name} \
--lr_layer_wise ${lr_layer_wise} --lr 1 \
--l1_dist_to_block 12 --l1_dist_w ${l1_dist_w} \
--resume local --skip_tune True \
--data ${DATA_PATH} \
--NODE_NUM 10 --RANK ${RANK} --MASTER ${MASTER}
```

Linear eval (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/lev2tgi2u0s7exo/AADcrb6F86zO8EG1yBFjPYR-a?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_large_imagenet_lr1B2080E300_graftFmae_pretrain_vit_large_lrLayerW1.5e-5,1.5e-5,1.5e-4_l1W1e-5To12

bash cmds/shell_scripts/tunePretrained_sweep.sh --pretrain_name ${pretrain_name} \
--dataset imagenet --batch_size 4096 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
--save_dir ./checkpoints_tune --arch vit_large -g 8 -p 4833 --data ${DATA_PATH} --resume local
```

Finetune (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/t44wlpfs5ovs0uq/AAA8wxR0J9LfG7j21IpgiIWIa?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_large_imagenet_lr1B2080E300_graftFmae_pretrain_vit_large_lrLayerW1.5e-5,1.5e-5,1.5e-4_l1W1e-5To12

bash cmds/shell_scripts/tunePretrained_mae.sh --pretrain_name ${pretrain_name} --dataset imagenet \
--batch_size 512 --accum_iter 2 --lr 5e-4 --layer_decay 0.8 --drop_path 0.2 --epochs 50 \
 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
--save_dir ./checkpoints_tune --arch vit_large -g 8 -p 4833 \
--data ${DATA_PATH} --resume local
```

1% Few shot (train with 1 1080Ti), [Checkpoint](https://www.dropbox.com/sh/q78xnvsbyyguark/AAASc9ZQtVTT1V0grZuFWwINa?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_large_imagenet_lr1B2080E300_graftFmae_pretrain_vit_large_lrLayerW1.5e-5,1.5e-5,1.5e-4_l1W1e-5To12

save_dir="."

bash cmds/shell_scripts/tunePretrained_sweep.sh --pretrain_name ${pretrain_name} --dataset imagenet --batch_size 4096 \
--pretrain ${save_dir}/checkpoints/${pretrain_name}/checkpoint_final.pth.tar  \
--save_dir ${save_dir}/checkpoints_tune --arch vit_large -g 1 -p 5893 --data ${DATA_PATH} \
--customSplit imagenet_1percent --customSplitName 1perc --batch_size 256 --logstic_reg True
```

10% Few shot (train with 8 V100s), [Checkpoint](https://www.dropbox.com/sh/z9s7k1m3tocu5qv/AAAmNSo9XLXbKeuW-5vK8EKIa?dl=0)
```shell
DATA_PATH="YourImageNetPath"
pretrain_name=vit_large_imagenet_lr1B2080E300_graftFmae_pretrain_vit_large_lrLayerW1.5e-5,1.5e-5,1.5e-4_l1W1e-5To12

bash cmds/shell_scripts/tunePretrained_mae.sh --pretrain_name ${pretrain_name} --dataset imagenet \
--batch_size 512 --accum_iter 2 --lr 3e-5 --layer_decay 0.65 --epochs 400 --tuneFromFirstFC True \
 --pretrain ./checkpoints/${pretrain_name}/checkpoint_final.pth.tar \
 --customSplit "imagenet_10percent" --customSplitName "10perc" --test_interval 10 \
--save_dir ./checkpoints_tune --arch vit_large -g 8 -p 4833 \
--data ${DATA_PATH} --resume local
```