#!/bin/bash

#SBATCH --partition=GPUampere
#SBATCH --time=200:00:00
#SBATCH --job-name=VS-DDPM_Task1_SwinVIT_MAE_192_192_32_distinctNorm_region_HN_TH_AB_linear_-1000_3000_DA_0.5_pen_var_random_T
#SBATCH --output=sbatch_out_final/VS-DDPM_Task1_SwinVIT_MAE_192_192_32_distinctNorm_region_HN_TH_AB_linear_-1000_3000_DA_0.5_pen_var_random_T_%J.txt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

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
    --n_epochs 3000 \
    --pacience 3000 \
    --val_interval 3000 \
    --dropout 0.2 \
    --lr 0.0001 \
    --clip_min_ct -1000 \
    --clip_max_ct 3000 \
    --data_norm_ct ScaleIntensityRanged \
    --data_norm_mri NormalizeIntensityd_Scaled \
    --mri_clip_percentile \
    --prob 0.5 \
    --verbose \
    --shuffle \
    --noise_schedule linear \
    --random_T_steps \
    --use_cosine_scheduler \
    --resume /projects/nian/synthrad2025/results/MC-IDDPM/VS-DDPM_Task1_2_3000_timestep__patchsize_1_SwinVIT_MAE_192_192_32_distinctNorm_region_HN_TH_AB_linear_DA_0.5_random_T_CTminmax-1000_3000/wandb/run-20260102_085040-wrlbfzmh/files/model/A_to_B_model_1699.pt
    # --penalize_high_variance \
    #--add_train_metric \
    # --add_train_metric_weight \
    # /projects/nian/synthrad2025/results/MC-IDDPM/VS-DDPM_Task1_2_3000_timestep__patchsize_2_SwinVIT_MSE_MAE_SSIM_128_128_32_distinctNorm_region_HN_TH_AB_linear_DA_0.5_pen_var_random_T_CTminmax-1000_1600/wandb/run-20251024_173146-v46dozdf/files/model/A_to_B_model_1799.pt
    #
    
