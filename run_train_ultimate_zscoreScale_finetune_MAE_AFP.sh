#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=100:00:00
#SBATCH --job-name=VS-DDPM_Task1_2_1000_timestep__patchsize_1_SwinVIT_MAE_AFP_192_192_32_distinctNorm_region_HN_TH_AB_linear_random_T_CTminmax-1000_1600_finetune
#SBATCH --output=sbatch_out_final/VS-DDPM_Task1_2_1000_timestep__patchsize_1_SwinVIT_MAE_AFP_192_192_32_distinctNorm_region_HN_TH_AB_linear_random_T_CTminmax-1000_1600_finetune_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=g1-1

# penalize_high_variance Added to avoid huge variance in the model output, which can lead to NaN values during training.

python train_mc_IDDPM.py \
    --network SwinVIT \
    --batch_size_train 2 \
    --patch_num 1 \
    --num_workers 8 \
    --patch_size 192 192 32 \
    --dataset_path /projects/nian/synthrad2025/Dataset/DataSet_Registered_2.0 \
    --region HN TH AB \
    --cache_rate 1.0 \
    --train_metric MAE \
    --task Task1 \
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
    --data_norm_mri NormalizeIntensityd_Scaled \
    --mri_clip_percentile \
    --prob 0 \
    --add_train_metric AFP \
    --add_train_metric_weight 0.2 \
    --verbose \
    --shuffle \
    --noise_schedule linear \
    --random_T_steps \
    --use_cosine_scheduler \
    --resume /projects/nian/synthrad2025/results/MC-IDDPM/VS-DDPM_Task1_2_1000_timestep__patchsize_1_SwinVIT_MAE_AFP_192_192_32_distinctNorm_region_HN_TH_AB_linear_random_T_CTminmax-1000_1600_finetune/wandb/run-20260114_042949-qu0hsmci/files/model/A_to_B_model_699.pt \
    --finetune 
    
    ##--penalize_high_variance \
