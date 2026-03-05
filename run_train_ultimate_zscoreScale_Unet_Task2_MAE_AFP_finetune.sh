#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=720:00:00
#SBATCH --job-name=VS-DDPM_Task2_2_1000_timestep__patchsize_2_Unet_MAE_AFP_128_128_32_ScaleIntensityRanged_region_HN_TH_AB_linear_DA_0.5_pen_var_random_T_CTminmax-1000_1600_finetune
#SBATCH --output=sbatch_out_final/VS-DDPM_Task2_2_1000_timestep__patchsize_2_Unet_MAE_AFP_128_128_32_ScaleIntensityRanged_region_HN_TH_AB_linear_DA_0.5_pen_var_random_T_CTminmax-1000_1600_finetune_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Nice shape to use -> 128 128 32 (32 because only two cases have Z smaller than 32. 128 because 256 is too big) 
# Training script for Synthetic-CT generation from MRI
# --network SwinVIT, SwinUNETR_vit, SwinUNETR, nnUNet
# --region HN TH AB

# penalize_high_variance Added to avoid huge variance in the model output, which can lead to NaN values during training.

## TODO also train without     --random_T_steps and --use_cosine_scheduler for timestep_respacing "25"
python train_mc_IDDPM.py \
    --network Unet \
    --batch_size_train 2 \
    --patch_num 2 \
    --num_workers 8 \
    --patch_size 128 128 32 \
    --dataset_path /projects/nian/synthrad2025/Dataset/DataSet_Registered_2.0 \
    --region HN TH AB \
    --cache_rate 1.0 \
    --train_metric MAE \
    --task Task2 \
    --timestep_respacing "" \
    --timestep_respacing_val "25" \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --overlap_mode 'constant' \
    --path_checkpoint ../../results/MC-IDDPM \
    --n_epochs 1000 \
    --pacience 1000 \
    --val_interval 1000 \
    --dropout 0.0 \
    --lr 0.00002 \
    --clip_min_ct -1000 \
    --clip_max_ct 1600 \
    --data_norm_ct ScaleIntensityRanged \
    --prob 0 \
    --add_train_metric AFP \
    --add_train_metric_weight 0.2 \
    --verbose \
    --shuffle \
    --noise_schedule linear \
    --random_T_steps \
    --use_cosine_scheduler \
    --resume /projects/nian/synthrad2025/results/MC-IDDPM/VS-DDPM_Task2_2_1600_timestep__patchsize_2_Unet_MAE_128_128_32_ScaleIntensityRanged_region_HN_TH_AB_linear_DA_0.5_pen_var_random_T_CTminmax-1000_1600/wandb/run-20251229_153832-lt4vh3l4/files/model/A_to_B_model_999.pt \
    --finetune 
