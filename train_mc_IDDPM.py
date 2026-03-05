print("Start importing...")
import wandb
import os
import argparse
import random
import monai
import json
import sys
import warnings
sys.path.append("..")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from os.path import join
from os import listdir
from os.path import isdir
from itertools import islice
import time
import torch
from diffusion.Create_diffusion import *
from diffusion.resampler import *
import torch.nn.functional as F
from MonaiDataLoader import MonaiDataLoader
from monai.metrics import PSNRMetric
from monai.losses.ssim_loss import SSIMLoss
import nibabel as nib
from tqdm.auto import tqdm
from network.Diffusion_model_transformer import *
from network.Diffusion_model_Unet import *
from network.Pre_trained_networks import load_pretrained_swinvit, load_pretrained_SwinUNETR, load_pretrained_TotalSegmentator, freeze_layers
from monai.inferers import SlidingWindowInferer
import SimpleITK as sitk
from utils.totalSegmentatorLoss import TotalSegmentatorLoss 
from utils.EMASmoother import EMASmoother
from utils.AFP_loss.AFP import AFP
import re
from monai.metrics import PSNRMetric
from monai.metrics.regression import SSIMMetric
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import nibabel as nib

#torch.autograd.set_detect_anomaly(True) # TODO remove
#print("Remove set_detect_anomaly!")
print("Finished importing.")

def set_complete_seed(seed):
    """
    Sets the seed for reproducibility across multiple libraries and environments.

    Args:
        seed (int): The seed value to be set.
    """
    # Python random module
    random.seed(seed)

    # Numpy random generator
    np.random.seed(seed)

    # PyTorch random generator for CPU
    torch.manual_seed(seed)

    # PyTorch random generator for all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in PyTorch operations on CUDA
        # I will swap to make the training faster
        torch.backends.cudnn.deterministic = False # True for deterministic
        torch.backends.cudnn.benchmark = True # False for deterministic

    # MONAI deterministic behavior
    #monai.utils.set_determinism(seed=seed) # Uncomment for deterministic

    # Set environment variables (may help in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # For newer CUDA versions # Uncomment for deterministic

    print(f"Complete seed set to {seed} for reproducibility across libraries and environment.")

def get_data_list(dataset_path, task_datasplit_json, task, region, args):
    """
    Generate data lists for training and validation based on the dataset path, task, and regions.

    Args:
        dataset_path (str): Path to the dataset root directory.
        task_datasplit_json (str): Path to the JSON file containing data splits.
        task (str): Task identifier, either "Task1" or "Task2".
        region (list[str], optional): List of regions to include. Defaults to ["AB", "HN", "TH"].

    Returns:
        tuple: A tuple containing:
            - data_list_task_train (list[dict]): List of dictionaries for training data with keys 
              corresponding to modalities (e.g., "ct", "mask", "mri" or "cbct").
            - data_list_task_val (list[dict]): List of dictionaries for validation data with keys 
              corresponding to modalities (e.g., "ct", "mask", "mri" or "cbct").
    """
    # Load the JSON file containing data splits
    with open(task_datasplit_json, "r") as f:
        task_data_split = json.load(f)

    # Collect all training and validation cases
    all_train_cases = []
    all_val_cases = []
    for region_name in region:
        train_cases = task_data_split[region_name]['train']
        val_cases = task_data_split[region_name]['val']
        all_train_cases.extend([join(dataset_path, case) for case in train_cases])
        all_val_cases.extend([join(dataset_path, case) for case in val_cases])

    # Initialize data lists
    data_list_task_train = []
    data_list_task_val = []
    print(f"all_train_cases: {[all_train_cases[0]]}")
    # Populate data lists based on the task
    if task == "Task1":
        for case in all_train_cases:
            seg_case = re.sub(r"synthRAD2025_Task1_Train(_D)?", "Task1_seg", case)
            data_list_task_train.append({
                "ct": join(case, "ct.mha"),
                "mask": join(case, "mask.mha"),
                "mri": join(case, "mr.mha"),
                "seg": join(seg_case, "pred_seg.mha") 
            })
        for case in all_val_cases:
            data_list_task_val.append({
                "ct": join(case, "ct.mha"),
                "mask": join(case, "mask.mha"),
                "mri": join(case, "mr.mha"),
                "seg": join(case, "pred_seg.mha") 
            })
    elif task == "Task2":
        for case in all_train_cases:
            data_list_task_train.append({
                "ct": join(case, "ct.mha"),
                "mask": join(case, "mask.mha"),
                "cbct": join(case, "cbct.mha"),
            })
        for case in all_val_cases:
            data_list_task_val.append({
                "ct": join(case, "ct.mha"),
                "mask": join(case, "mask.mha"),
                "cbct": join(case, "cbct.mha")
            })
    else:
        raise ValueError("task should be either 'Task1' or 'Task2'")

    return data_list_task_train, data_list_task_val

def get_dataloader(data_list_task_train, data_list_task_val, task, args):
    """
    Creates and returns the training and validation data loaders.

    Args:
        data_list_task_train (list): List of training cases.
        data_list_task_val (list): List of validation cases.

    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    if task=="Task1":
        args.key_in = "mri"
    elif task=="Task2":
        args.key_in = "cbct"
    else:
        raise ValueError("task should be either 'Task1' or 'Task2'")
    if args.region_clip:
        if len(args.region) > 1 and args.region_clip:
            raise ValueError("'region' should be either (only one) HN, TH, or AB to use 'region_clip'")
        else:
            print("WARNING: 'clip_min_ct' and 'clip_max_ct' values will be ignored when 'region_clip' is enabled.")
    
    print(f"Shuffle data: {args.shuffle}")
    data_loader = MonaiDataLoader(
        data_list_task_train, 
        data_list_task_val, 
        spatial_size=args.patch_size, 
        patch_num=args.patch_num, 
        key_in=args.key_in, 
        key_out="ct", 
        key_mask="mask", 
        cache_rate=args.cache_rate, 
        batch_size=args.batch_size_train, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers,
        region=args.region,
        region_clip=args.region_clip,
        a_min_ct=args.clip_min_ct, 
        a_max_ct=args.clip_max_ct,
        data_norm_ct=args.data_norm_ct,
        data_norm_mri=args.data_norm_mri,
        for_totalsegmentator=("DSC" in args.add_train_metric),
        prob=args.prob,
        mri_clip_percentile=args.mri_clip_percentile
    )
    print(f"('DSC' in args.add_train_metric): {('DSC' in args.add_train_metric)}")
    train_dataloader, val_dataloader, train_transforms, val_transforms = data_loader.get_dataloader()

    return train_dataloader, val_dataloader, train_transforms, val_transforms

def get_diffusion(timestep_respacing, timestep_respacing_val, args):
    """
    Define the gaussian diffusion scheduler for training.
    These three parameters: training steps number, learning variance or not (using improved DDPM or original DDPM), and inference 
    timesteps number (only effective when using improved DDPM)
    In:
        timestep_respacing: Used for training
        timestep_respacing_val: Used mainly for inference to reduce the number of steps.
    Out:
        train_diffusion, val_diffusion, schedule_sampler: Diffusion models and the schedule sampler
        # val_diffusion has less time steps for inference
    """
    # Hard coded parameters
    #  
    sigma_small=False # Doesn't make a difference with learn_sigma=True
    noise_schedule=args.noise_schedule
    use_kl=False
    predict_xstart=True # IDDPM style
    rescale_timesteps=True # IDDPM style
    rescale_learned_sigmas=True # IDDPM style # loss_type = gd.LossType.RESCALED_MSE
    diffusion_steps=1000 # 
    learn_sigma=True # ModelVarType.LEARNED_RANGE

    print(f"Using {noise_schedule} noise schedule with {timestep_respacing} steps for training and {timestep_respacing_val} steps for validation.")
    train_diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    val_diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing_val,
    )

    schedule_sampler = UniformSampler(train_diffusion)
    return train_diffusion, val_diffusion, schedule_sampler

def get_SwinVITmodel(args):
    """
    Build the MC-IDDPM network
    Here enter your network parameters:num_channels means the initial channels in each block,
    channel_mult means the multipliers of the channels (in this case, 128,128,256,256,512,512 for the first to the sixth block),
    attention_resolution means we use the transformer blocks in the third to the sixth block
    number of heads, window size in each transformer block
    """
    # Hard coded arguments
    num_channels=64
    attention_resolutions="32,16,8"
    channel_mult = (1, 2, 3, 4)
    num_heads=[4,4,8,16]
    window_size = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
    num_res_blocks = [2,2,2,2]
    sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),
    use_scale_shift_norm = True
    resblock_updown = False
    dropout = args.dropout
    use_checkpoint=False

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    
    A_to_B_model = SwinVITModel(
            image_size=args.patch_size,
            in_channels=2,
            model_channels=num_channels,
            out_channels=2,
            dims=3,
            sample_kernel = sample_kernel,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=None,
            use_checkpoint=use_checkpoint,
            use_fp16=False,
            num_heads=num_heads,
            window_size = window_size,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=False,
        )
    return A_to_B_model

def get_heavy_Unet(args):
    num_channels = 64
    num_res_blocks = [2,2,2,2,2]
    attention_resolutions="32,16,8"
    sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),
    channel_mult = (1,2,3,4) 
    use_scale_shift_norm = True
    class_cond=False

    num_head_channels = 64

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    A_to_B_model = UNetModel(
            img_size = args.patch_size,
            image_size= args.patch_size[0],
            in_channels=2,
            model_channels=num_channels,
            out_channels=2,
            dims = 3,
            num_res_blocks=num_res_blocks[0],
            attention_resolutions=tuple(attention_ds),
            dropout=args.dropout,
            sample_kernel=sample_kernel,
            channel_mult=channel_mult,
            num_classes=(128 if class_cond else None),
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=False, # TODO if I get problems loading the network, this might be the issue...
            use_new_attention_order=False,
        )

    return A_to_B_model

def get_Unet(args):
    num_channels = 64
    num_res_blocks = [2,2,2,2,2]
    attention_resolutions="16,8"
    sample_kernel=([2,2,1],[2,2,1],[2,2,2],[2,2,2],[2,2,2]),
    channel_mult = (1,2,2,4,4) 
    use_scale_shift_norm = True
    class_cond=False

    num_head_channels = -1

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    A_to_B_model = UNetModel(
            img_size = args.patch_size,
            image_size= args.patch_size[0],
            in_channels=2,
            model_channels=num_channels,
            out_channels=2,
            dims = 3,
            num_res_blocks=num_res_blocks[0],
            attention_resolutions=tuple(attention_ds),
            dropout=args.dropout,
            sample_kernel=sample_kernel,
            channel_mult=channel_mult,
            num_classes=(128 if class_cond else None),
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=False, # TODO if I get problems loading the network, this might be the issue...
            use_new_attention_order=False,
        )

    return A_to_B_model

def get_Unet_lighter(args):
    num_channels = 32
    num_res_blocks = 2
    attention_resolutions="16,8"
    sample_kernel=([2,2,1],[2,2,1],[2,2,2],[2,2,2],[2,2,2]),
    channel_mult = (1,2,3,4,4) 
    use_scale_shift_norm = True
    class_cond=False

    num_head_channels = -1

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(int(res))
    A_to_B_model = UNetModel(
            img_size = args.patch_size,
            image_size= args.patch_size[0],
            in_channels=2,
            model_channels=num_channels,
            out_channels=2,
            dims = 3,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=args.dropout,
            sample_kernel=sample_kernel,
            channel_mult=channel_mult,
            conv_resample=False,
            use_avg_pool_down=True, # Only if conv_resample=False
            use_trilinear_up=True, # Only if conv_resample=False
            num_classes=(128 if class_cond else None),
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=False,
            use_new_attention_order=False,
        )

    return A_to_B_model

def get_model(device, args):
    """
    Initializes and returns a model based on the specified network type.

    Args:
        device (torch.device): The device on which the model will be loaded 
            (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The initialized model corresponding to the specified 
            network type, along with filtered_dict and not_matching_keys 
            (if applicable).

    Raises:
        ValueError: If the specified network type is not recognized. Valid 
            options are 'SwinVIT', 'SwinUNETR_vit', 'SwinUNETR', or 'nnUNet'.

    Note:
        The network type is determined by the global `args.network` variable.
        Ensure that `args.network` is set to one of the valid options before 
        calling this function.
    """
    filtered_dict, not_matching_keys = None, None
    if args.network=="SwinVIT":
        A_to_B_model = get_SwinVITmodel(args)
    
    elif args.network=="Unet":
        A_to_B_model = get_Unet(args)
    
    elif args.network=="Unet_heavy":
        A_to_B_model = get_heavy_Unet(args)
    
    elif args.network=="Unet_lighter":
        A_to_B_model = get_Unet_lighter(args)

    elif args.network=="SwinUNETR_vit":
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_swinvit(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained, 
                                                            img_size=args.patch_size, 
                                                            in_channels=2, 
                                                            out_channels=2, 
                                                            feature_size=48, 
                                                            use_checkpoint=True, 
                                                            verbose=args.verbose
                                                            )
    elif args.network=="SwinUNETR":
        print(f"Using SwinUNETR")
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_SwinUNETR(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained, 
                                                            img_size=args.patch_size, 
                                                            in_channels=2, 
                                                            out_channels=2, 
                                                            feature_size=48, 
                                                            use_checkpoint=True, 
                                                            verbose=args.verbose
                                                            )
    elif args.network=="nnUNet":
        print(f"Using nnUNet")
        A_to_B_model, filtered_dict, not_matching_keys = load_pretrained_TotalSegmentator(
                                                            ckpt_path=args.path_pretrained, 
                                                            load_weights=args.load_pretrained,
                                                            verbose=args.verbose
                                                            )
    else:
        raise ValueError(f"Unknown network type: {args.network}" \
        "Please choose 'SwinVIT' (for original implementation), 'Unet', "
        " 'SwinUNETR_vit' (for loading vit pre trained weights),"
        " 'SwinUNETR' (to train from scratch or loading pre-trained weights from SwinUNETR),"
        " or 'nnUNet' (to train from scratch or loading pre-trained weights from TotalSegmentator).")

    A_to_B_model = A_to_B_model.to(device)
    return A_to_B_model, filtered_dict, not_matching_keys

def get_additional_losses(target, model_output, args, loss_kwargs):
    """
    Calculates weighted loss values using multiple metrics.

    For each metric listed in args.add_train_metric, this function applies the corresponding
    loss function from args.more_train_metric to the model output and target, multiplies it 
    by the given weight from args.add_train_metric_weight, and stores the result.

    Parameters:
        target: Ground truth values.
        model_output: Model predictions.
        args: An object with:
            - add_train_metric: List of metric names.
            - add_train_metric_weight: List of weights for each metric.
            - more_train_metric: Dictionary of loss functions.

    Returns:
        A dictionary of weighted loss values for each metric.
    """
    all_losses = {}
    for idx, new_metric in enumerate(args.add_train_metric):
        loss_fun_now = args.more_train_metric[new_metric]
        if new_metric == "DSC":
            loss_value_now = loss_fun_now(
                traintarget_full_res= loss_kwargs["traintarget_full_res"], 
                predicted_patch=model_output, 
                gt_seg= loss_kwargs["gt_seg"], 
                limits= loss_kwargs["limits"])
        if new_metric == "SSIM":
            # Clip intensity values to ensure data within range (might be warmfull for gradients!)
            model_output_here = torch.clamp(model_output, min=args.min_clip, max=args.max_clip) - args.min_clip
            target_here = torch.clamp(target, min=args.min_clip, max=args.max_clip) - args.min_clip
            # Compute SSIM loss
            loss_value_now = loss_fun_now(model_output_here, target_here)
            # Ensure it's not negative
            loss_value_now = torch.clamp(loss_value_now, min=0, max=2)
        if new_metric == "SSIM_tahn":
            # Use tahn for stability
            model_output_here = torch.tanh(model_output) # ensure between -1:1
            target_here = torch.tanh(target) # ensure between -1:1
            # Compute SSIM loss
            loss_value_now = loss_fun_now(model_output_here, target_here)
        if new_metric == "AFP":
            # Pre-processing is done within the AFP object
            loss_value_now, output_x, output_y = loss_fun_now(target, model_output, preprocess=True)
        else:
            loss_value_now = loss_fun_now(model_output, target)

        all_losses[new_metric] = loss_value_now.mean() * float(args.add_train_metric_weight[idx])
    return all_losses

def random_foreground_crop_batch(
    ct_fullres: torch.Tensor,      # (B, C, H, W, D)
    mri_fullres: torch.Tensor,     # (B, C, H, W, D)
    mask: torch.Tensor,            # (B, C, H, W, D)
    crop_size=(128, 128, 32)
    ):
    """
    Perform random foreground cropping over a batch.
    Returns cropped volumes and coordinates for each item in batch.
    """
    B, C, H, W, D = ct_fullres.shape
    crop_H, crop_W, crop_D = crop_size

    cropped_ct = torch.empty((B, C, crop_H, crop_W, crop_D), dtype=ct_fullres.dtype, device=ct_fullres.device)
    cropped_mri = torch.empty_like(cropped_ct)
    crop_coords = torch.empty((B, 6), dtype=torch.int32, device=ct_fullres.device)

    mask_squeezed = mask[:, 0]  # Assuming C=1 for the mask.

    for b in range(B):
        fg_indices = (mask_squeezed[b] == 1).nonzero(as_tuple=False)
        if fg_indices.size(0) == 0:
            raise ValueError(f"Sample {b} contains no foreground voxels.")

        idx = torch.randint(0, fg_indices.size(0), (1,), device=ct_fullres.device)
        center_y, center_x, center_z = fg_indices[idx].squeeze(0)

        start_y = torch.clamp(center_y - crop_H // 2, min=0, max=H - crop_H)
        start_x = torch.clamp(center_x - crop_W // 2, min=0, max=W - crop_W)
        start_z = torch.clamp(center_z - crop_D // 2, min=0, max=D - crop_D)

        end_y = start_y + crop_H
        end_x = start_x + crop_W
        end_z = start_z + crop_D

        cropped_ct[b] = ct_fullres[b, :, start_y:end_y, start_x:end_x, start_z:end_z]
        cropped_mri[b] = mri_fullres[b, :, start_y:end_y, start_x:end_x, start_z:end_z]
        crop_coords[b] = torch.tensor([start_y, start_x, start_z, end_y, end_x, end_z], dtype=torch.int32, device=ct_fullres.device)
        return cropped_ct, cropped_mri, crop_coords

def train(model, optimizer, data_loader1, scaler, for_totalsegmentator, args):
    """
    Training function.
    Called once per epoch.
    In:
        model: model weights
        optimizer: optimizer for model weight update
        data_loader1: dataloader from get_dataloader
    Out:
        Average loss
    """
    #1: set the model to training mode
    model.train()
    A_to_B_losses = {'loss': []} # Initialize the loss dictionary
    total_time = 0

    for idx, new_metric in enumerate(args.add_train_metric):
        A_to_B_losses[new_metric] = []

    #2: Loop the whole dataset
    # x1 is the input data (MRI)
    # x2 is the target data (CT)

    # Initialize the tqdm progress bar once
    progress_bar = tqdm(data_loader1, desc="Processing Batches", total=len(data_loader1))

    train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
        timestep_respacing=str(args.timestep_respacing),
        timestep_respacing_val=args.timestep_respacing_val,
        args=args
        )

    for i, batch in enumerate(progress_bar):  # Use the progress_bar here
        if args.finetune and args.random_T_steps:
            possible_T = [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
            args.timestep_respacing = random.choice(possible_T)
   
            train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
                timestep_respacing=str(args.timestep_respacing),
                timestep_respacing_val=args.timestep_respacing_val,
                args=args
                )
        if for_totalsegmentator:
            traincondition_full_res = batch[args.key_in].to(device)
            traintarget_full_res = batch['ct'].to(device)
            mask = batch['mask'].to(device)
            seg = batch['seg_downsample'].to(device)
            traintarget, traincondition, crop_coords = random_foreground_crop_batch(
                traintarget_full_res,
                traincondition_full_res,
                mask,
                crop_size=args.patch_size
            )
            loss_kwargs = {
                "gt_seg": seg,
                "limits": crop_coords,
                "traintarget_full_res": traintarget_full_res
            }
        else:
            traincondition = batch[args.key_in].to(device)
            traintarget = batch['ct'].to(device)
            loss_kwargs = {}
            """
            # TODO remove this save
            # Save each sample in the batch individually
            # Move tensor back to CPU and convert to numpy
            output_dir = "/projects/nian/synthrad2025/trash"
            traincondition_np = traincondition.cpu().numpy()
            print(f"traincondition_np: {traincondition_np.shape}")
            for batch_idx in range(traincondition_np.shape[0]):
                sample = traincondition_np[batch_idx]  # shape [C, H, W, D] or [H, W, D] depending on your data

                # If your data has channels, you might want to remove singleton dimension
                if sample.shape[0] == 1:
                    sample = sample[0]

                # Create a NIfTI image
                nii_img = nib.Nifti1Image(sample, affine=np.eye(4))  # Using identity affine for simplicity

                # Save the NIfTI file
                filename = os.path.join(output_dir, f"{i}_cond_batch_sample{batch_idx}.nii.gz")
                nib.save(nii_img, filename)

                print(f"Saved {filename}")

            traintarget_np = traintarget.cpu().numpy()
            for batch_idx in range(traintarget_np.shape[0]):
                sample = traintarget_np[batch_idx]  # shape [C, H, W, D] or [H, W, D] depending on your data

                # If your data has channels, you might want to remove singleton dimension
                if sample.shape[0] == 1:
                    sample = sample[0]

                # Create a NIfTI image
                nii_img = nib.Nifti1Image(sample, affine=np.eye(4))  # Using identity affine for simplicity

                # Save the NIfTI file
                filename = os.path.join(output_dir, f"{i}_target_batch_sample{batch_idx}.nii.gz")
                nib.save(nii_img, filename)

                print(f"Saved {filename}")

            """
        if torch.isnan(traincondition).any() or torch.isnan(traintarget).any(): 
            print(f"NaN detected in input data at batch {i}")
            continue 

        # Reshape the target and input images based on args.patch_size
        # Not needed, done by the dataloader

        # Extract random timestep for training
        t, weights = schedule_sampler.sample(traincondition.shape[0], device)

        # Optimize the TDM network
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # Compute the losses
            all_loss, target, model_output = train_diffusion.training_losses(A_to_B_model, traintarget, traincondition, t, train_metric=args.train_metric, penalize_high_variance=args.penalize_high_variance)
            A_to_B_loss = (all_loss["loss"] * weights).mean() # Loss from the train_metric (default is MSE loss)

            # Check if the model's prediction itself is unstable
            if not torch.isfinite(model_output).all():
                print(f"Batch {i}: `model_output` (predicted x_start) is NaN/Inf. Skipping batch.")
                # This 'continue' skips .backward() and .step(),
                # preventing the crash and moving to the next batch.
                del all_loss
                del target
                del model_output
                del A_to_B_loss
                continue 

            # Add the weighted losses from additional metrics
            # If none are provided, this will be skipped
            addicional_losses = get_additional_losses(target, model_output, args, loss_kwargs)
            for add_loss in addicional_losses:
                A_to_B_loss += addicional_losses[add_loss]
                A_to_B_losses[add_loss].append(addicional_losses[add_loss].mean().detach().cpu().numpy())
                wandb.log({add_loss: addicional_losses[add_loss].item()}) # wandb save

            wandb.log({"mse": (all_loss["mse"] * weights).mean().item()}) # wandb save
            wandb.log({"vb_loss": (all_loss["vb"] * weights).mean().item()}) # wandb save

            if args.penalize_high_variance:
                wandb.log({"loss_high_variance": (all_loss["loss_high_variance"] * weights).mean().item()}) # wandb save
            wandb.log({"Loss": A_to_B_loss.item()}) # wandb save

            # Check for NaN values in all_loss
            if torch.isnan(A_to_B_loss): 
                print("NaN detected in loss, skipping this batch.")
                del all_loss
                del A_to_B_loss
                continue

            # Append the loss value for tracking
            A_to_B_losses["loss"].append(all_loss["loss"].mean().detach().cpu().numpy())

        # Update tqdm with loss info
        progress_bar.set_postfix(base_loss=all_loss["loss"].mean().detach().cpu().numpy())


        # 1) Scale and backward as before
        scaler.scale(A_to_B_loss).backward()

        # 2) Unscale the gradients back to FP32
        scaler.unscale_(optimizer)

        # 5) Guard: check all grads are finite
        all_finite = True
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                print(f"Batch {i}: NON‑FINITE gradient in {name}, skipping optimizer.step()")
                all_finite = False
                break 

        # 6) Guard: check weights are still finite
        if all_finite:
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"Batch {i}: NON‑FINITE weight in {name}, skipping optimizer.step()")
                    all_finite = False
                    break 

        # 7) Only step & update scaler if everything is finite
        if all_finite:
            scaler.step(optimizer)
        else:
            # reset any corrupted state to avoid poisoning future steps
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state.get(p, {})
                    for k, v in state.items():
                        if torch.is_tensor(v) and not torch.isfinite(v).all():
                            optimizer.state[p][k] = torch.zeros_like(v)
            print(f"Batch {i}: optimizer.step() SKIPPED and state reset. Timesteps with error used: {t}")
            print(f"Batch {i}: Scaled loss = {scaler.get_scale() * A_to_B_loss.item():.6f}")

        # 8) Finally update the scaler (always)
        scaler.update()

    if np.isnan(A_to_B_losses["loss"]).any(): 
        print("NaN values detected in loss history. Check earlier stages.")

    #6: print out total time 
    print("Total time per sample is: "+str(time.time()-total_time))

    return A_to_B_losses, model, optimizer, scaler

def diffusion_sampling(condition, model):
    sampled_images = val_diffusion.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3],condition.shape[4]),
        condition=condition,
        clip_denoised=args.clip_denoised
        )
    return sampled_images

def pad_if_smaller_symmetric(in_tensor, patch_size):
    """
    Pads a 5D tensor symmetrically along its depth, height, and width dimensions 
    if any of these dimensions are smaller than the corresponding patch size.

    Parameters:
        in_tensor (torch.Tensor): The input tensor with shape (B, C, D, H, W), 
                                  where B is the batch size, C is the number of channels, 
                                  D is the depth, H is the height, and W is the width.
        patch_size (tuple of int): A tuple (D_patch, H_patch, W_patch) specifying the 
                                   minimum required size for the depth, height, and width 
                                   dimensions of the tensor.

    Returns:
        tuple:
            - torch.Tensor: The padded tensor with the same batch size and channels, 
                            but with depth, height, and width dimensions padded symmetrically 
                            to meet or exceed the specified patch size.
            - tuple of int: A tuple (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2) 
                            representing the amount of padding applied to each side 
                            of the width, height, and depth dimensions, respectively.

    """
    pad_d = max(patch_size[0] - in_tensor.shape[2], 0)
    pad_h = max(patch_size[1] - in_tensor.shape[3], 0)
    pad_w = max(patch_size[2] - in_tensor.shape[4], 0)

    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2

    # F.pad expects (W_before, W_after, H_before, H_after, D_before, D_after)
    padding = (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2)

    return F.pad(in_tensor, padding), padding

def unpad_if_smaller_symmetric(in_tensor, padding):
    """
    Removes symmetric padding from a tensor if padding was applied.

    This function removes padding from the input tensor along the depth, height, 
    and width dimensions based on the specified padding values. The padding is 
    assumed to be symmetric, and the function ensures that the tensor is sliced 
    correctly even if some padding values are zero.

    Args:
        in_tensor (torch.Tensor): The input tensor to unpad. Expected to have 
            dimensions (batch_size, channels, depth, height, width).
        padding (tuple): A tuple of six integers specifying the padding values 
            in the order (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2), where:
            - pad_w1, pad_w2: Padding on the width dimension (start, end).
            - pad_h1, pad_h2: Padding on the height dimension (start, end).
            - pad_d1, pad_d2: Padding on the depth dimension (start, end).

    Returns:
        torch.Tensor: The unpadded tensor with the same number of dimensions as 
        the input tensor but with reduced size along the depth, height, and 
        width dimensions if padding was applied.
    """
    pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2 = padding

    if pad_d1 + pad_d2 > 0:
        in_tensor = in_tensor[:, :, pad_d1:-pad_d2 if pad_d2 > 0 else None, :, :]
    if pad_h1 + pad_h2 > 0:
        in_tensor = in_tensor[:, :, :, pad_h1:-pad_h2 if pad_h2 > 0 else None, :]
    if pad_w1 + pad_w2 > 0:
        in_tensor = in_tensor[:, :, :, :, pad_w1:-pad_w2 if pad_w2 > 0 else None]
    return in_tensor

def evaluate(model, epoch, data_loader1, inferer, args):
    """
    Run the evaluate function will translate the input image to CT.
    The result will be saved in nii.gz format
    """
    model.eval()
    loss_all = []
    if args.data_norm_ct == 'NormalizeIntensityd':
        normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{args.task}_Train_normalization_stats_{args.clip_min_ct}_{args.clip_max_ct}.json"
        with open(normalization_stats_path, "r") as stats_file:
            normalization_stats = json.load(stats_file)
        ct_mean = normalization_stats.get("ct_mean")
        ct_std = normalization_stats.get("ct_std")


    if (epoch+1) % args.val_interval == 0: 
        with torch.no_grad():
            for i, batch in enumerate(tqdm(islice(data_loader1, 1), desc="1 Random cases on validation:", total=1)):
                x1 = batch[args.key_in].cuda()
                y1 = batch['ct'].cuda()
                

                x1_padded, x1_padding = pad_if_smaller_symmetric(x1, args.patch_size)
                y1_padded, y1_padding = pad_if_smaller_symmetric(y1, args.patch_size)
                
                # Save key_in
                key_in_image_batch = x1[0][0].cpu().numpy()  
                # Rescale the values from (-1, 1) to original
                #key_in_image = sitk.ReadImage(batch[f'{args.key_in}_meta_dict']['filename_or_obj'])
                #key_in_image_array = sitk.GetArrayFromImage(key_in_image)
                #max_value = key_in_image_array.max()
                #min_value = key_in_image_array.min()
                #nii_key_in_image = ((key_in_image_batch - (key_in_image_batch.min())) / (key_in_image_batch.max() - (key_in_image_batch.min()))) * (max_value - (-min_value)) + (-min_value)
                nib.save(nib.Nifti1Image(
                    key_in_image_batch,
                    affine=batch[f'{args.key_in}_meta_dict']['original_affine'].numpy()[0]),
                    f'{args.path_checkpoint}/scans/{epoch}_{i}_in.nii.gz'
                    )
                # Save CT
                ct_image_batch = y1[0][0].cpu().numpy()

                if args.data_norm_ct == 'ScaleIntensityRanged':
                    # Rescale the values from (-1, 1) to (-1000, 2000) 
                    # The predicted values should be able to 
                    nii_ct_image = ((ct_image_batch - (args.min_clip)) / (args.max_clip - (args.min_clip))) * (args.clip_max_ct - (args.clip_min_ct)) + (args.clip_min_ct)
                elif args.data_norm_ct == 'NormalizeIntensityd':
                    nii_ct_image = ct_image_batch * ct_std + ct_mean
                else:
                    raise ValueError("'data_norm' should be either ScaleIntensityRanged or NormalizeIntensityd")
                nib.save(nib.Nifti1Image(
                    nii_ct_image, 
                    affine=batch['ct_meta_dict']['original_affine'].numpy()[0]), 
                    f'{args.path_checkpoint}/scans/{epoch}_{i}_gt.nii.gz'
                    )
                
                # Prediction
                # condition is the input key_in
                # target is the target CT
                condition = x1_padded.to(device)
                target = y1.to(device)            
                # sampled_images is the synthetic CT
                with torch.amp.autocast("cuda"):
                    sampled_images = inferer(
                        condition,
                        diffusion_sampling,
                        model
                        )
                    if torch.isnan(sampled_images).any(): 
                        print(f"NaN detected in the sampled_images in inference.")
                    sampled_images = unpad_if_smaller_symmetric(sampled_images, x1_padding)
                # Save prediction
                if len(sampled_images.shape) == 5:
                    sampled_images = sampled_images[0][0]
                elif len(sampled_images.shape) == 4:
                    sampled_images = sampled_images[0]     
                predict_ct_image_batch = sampled_images.cpu().numpy() # Identity matrix as the affine transformation
                predict_ct_image_batch = np.clip(a=predict_ct_image_batch, a_min=args.min_clip, a_max=args.max_clip)
                
                if args.data_norm_ct == 'ScaleIntensityRanged':
                    # Rescale the values from e.g (-1, 1) to (-1000, 2000) 
                    # The predicted values should be able to create a correct scale from e.g -1 to 1 to translate into -1000 and 2000. 
                    # Some cases will not have e.g 2000 values
                    nii_pred_image = ((predict_ct_image_batch - (args.min_clip)) / (args.max_clip - (args.min_clip))) * (args.clip_max_ct - (args.clip_min_ct)) + (args.clip_min_ct)
                elif args.data_norm_ct == 'NormalizeIntensityd':
                    nii_pred_image = predict_ct_image_batch * ct_std + ct_mean
                else:
                    raise ValueError("'data_norm_ct' should be either ScaleIntensityRanged or NormalizeIntensityd")
                
                nib.save(nib.Nifti1Image(
                    nii_pred_image, 
                    affine=batch['ct_meta_dict']['original_affine'].numpy()[0]), 
                    f'{args.path_checkpoint}/scans/{epoch}_{i}_validation.nii.gz')
            
                if i >= 0: # Uses only 3 cases for validation
                    break
    with torch.no_grad():
        pbar = tqdm(data_loader1, desc="Validation")
        for i, batch in enumerate(pbar):
            condition = batch[f"{args.key_in}_crop"].cuda()
            target = batch['ct_crop'].cuda()
            
            if args.timestep_respacing_val=='':
                print("Using random steps t for validation.")
                # Predict patch
                t_all = []
                for _t_ in range(6):
                    t, weights = schedule_sampler.sample(condition.shape[0], device)
                    t_all.append(t)
            else:
                # Select timesteps for better validation
                # Fixed tensors
                t1 = torch.randint(1, 2, (condition.shape[0],), device=device)
            
                # Random tensors within specified ranges
                t2 = torch.randint(2, 6, (condition.shape[0],), device=device)   # 2 to 5
                t3 = torch.randint(5, 11, (condition.shape[0],), device=device)  # 5 to 10
                t4 = torch.randint(10, 16, (condition.shape[0],), device=device) # 10 to 15
                t5 = torch.randint(15, 21, (condition.shape[0],), device=device) # 15 to 20
                t6 = torch.randint(20, 25, (condition.shape[0],), device=device) # 20 to 24
                t_all = [t1,t2,t3,t4,t5,t6]
            for t in t_all:
                _, _, sampled_images = val_diffusion.training_losses(A_to_B_model, target, condition, t, train_metric=args.train_metric)
                    
                # Normalisation not needed
                MAE_value = eval_MAE(sampled_images, target)
                MSE_value = eval_MSE(sampled_images, target)

                # Clip intensity values to ensure data within range
                sampled_images_here = torch.clamp(sampled_images, min=args.min_clip, max=args.max_clip) - args.min_clip
                target_here = torch.clamp(target, min=args.min_clip, max=args.max_clip) - args.min_clip
                SSIM_value = eval_SSIM(sampled_images_here, target_here) 

                # Norm to compute PSNR
                sampled_images_here = sampled_images.clone()
                target_here = target.clone() 
                # Clip between min and max
                sampled_images_here = sampled_images_here.clamp(args.min_clip, args.max_clip)
                target_here = target_here.clamp(args.min_clip, args.max_clip)
                if args.data_norm_ct == 'ScaleIntensityRanged':
                    # Rescale the values from e.g. (-1, 1) to (-1000, 1600) 
                    # The predicted values should be able to 
                    sampled_images_here = ((sampled_images_here - (args.min_clip)) / (args.max_clip - (args.min_clip))) * (args.clip_max_ct - (args.clip_min_ct)) + (args.clip_min_ct)
                    target_here = ((target_here - (args.min_clip)) / (args.max_clip - (args.min_clip))) * (args.clip_max_ct - (args.clip_min_ct)) + (args.clip_min_ct)
                elif args.data_norm_ct == 'NormalizeIntensityd':
                    sampled_images_here = sampled_images_here * ct_std + ct_mean
                    target_here = target_here * ct_std + ct_mean
                # Making sure it is within the range and that the range is correct
                sampled_images_here = sampled_images_here.clamp(args.clip_min_ct, args.clip_max_ct)
                target_here = target_here.clamp(args.clip_min_ct, args.clip_max_ct)
                sampled_images_here = sampled_images_here-args.clip_min_ct
                target_here = target_here-args.clip_min_ct
                PSNR_value = eval_PSNR(sampled_images_here, target_here)
                # norm between 0 and 1
                #sampled_images_here = (sampled_images_here - sampled_images_here.min()) / (sampled_images_here.max() - sampled_images_here.min() + 1e-8)
                #target_here = (target_here - target_here.min()) / (target_here.max() - target_here.min() + 1e-8)
                #SSIM_value = eval_SSIM(sampled_images_here, target_here) 
               
                postfix_dict = {f"t_{i}": t[i].item() for i in range(len(t))}
                postfix_dict["mae_loss"] = f"{MAE_value.mean().cpu().numpy():.4f}"

                if torch.isnan(MAE_value).any(): 
                    print("NaN detected in validation loss. MAE_value.")
                if torch.isnan(MSE_value).any(): 
                    print("NaN detected in validation loss. MSE_value.")
                if torch.isnan(PSNR_value).any(): 
                    print("NaN detected in validation loss. PSNR_value.")
                if torch.isnan(SSIM_value).any(): 
                    print("NaN detected in validation loss. SSIM_value.")
                
                loss_all.append({
                    "MAE": MAE_value.cpu().numpy(),
                    "MSE": MSE_value.cpu().numpy(),
                    "PSNR": PSNR_value.cpu().numpy(),
                    "SSIM": SSIM_value.cpu().numpy(),
                })
        
            
        return loss_all

def warm_up_model(A_to_B_model, freeze_epochs, args):
        """
        Perform a warm-up phase for the given model by training it for a specified number of epochs 
        with frozen pre-trained parameters. The function logs the training loss for each warm-up epoch and saves 
        a checkpoint of the warmed-up model.

        Args:
            A_to_B_model (torch.nn.Module): The model to be warmed up.
            

        Returns:
            tuple: A tuple containing the warmed-up model (`torch.nn.Module`).

        Notes:
            - The number of warm-up epochs is determined by the `args.freeze_epochs` parameter.
            - Training is performed using the `train` function, which should be defined elsewhere.
            - The training loss for each warm-up epoch is logged using `wandb`.
            - A checkpoint of the warmed-up model is saved to the path specified by `args.path_checkpoint`.

        Checkpoint Contents:
            - `model_state_dict`: The state dictionary of the warmed-up model.
            - `optimizer_state_dict`: The state dictionary of the optimizer.
            - `epoch`: The current epoch number.
            - `best_loss`: 0

        Dependencies:
            - `np.nanmean`: Used to compute the average loss while ignoring NaN values.
            - `wandb.log`: Used for logging training metrics.
            - `torch.save`: Used for saving the model checkpoint.
            - `args`: A global variable containing configuration parameters such as `freeze_epochs` 
              and `path_checkpoint`.
            - `train_dataloader`: A global variable representing the training data loader.
            - `train`: A function that performs one epoch of training and returns the losses, updated 
              model, and optimizer.
        """
        #optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5, weight_decay = 1e-4)
        optimizer = torch.optim.AdamW(
            A_to_B_model.parameters(),  # frozen encoder params will be skipped
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        scaler = torch.amp.GradScaler("cuda")

        for warm_up_epoch in range (freeze_epochs):
            print('Warmup epoch:', warm_up_epoch)
            # Build the training function. Run the training function once = one epoch
            A_to_B_losses, A_to_B_model, optimizer, scaler = train(
                model=A_to_B_model, 
                optimizer=optimizer, 
                data_loader1=train_dataloader,
                scaler=scaler,
                )
            average_loss_train = np.nanmean(A_to_B_losses)
            wandb.log({"Loss warmup_epoch": average_loss_train.item(), "warmup_epoch": warm_up_epoch}) # wandb save
            print('Averaged warmup loss is: '+ str(average_loss_train))
        checkpoint = {
                    'model_state_dict': A_to_B_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': warm_up_epoch,
                    'best_loss': 0
                }
        torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_warmed_up.pt'))
        return A_to_B_model

def warmup_step(A_to_B_model, filtered_dict, not_matching_keys, args):
    """
    Perform a warm-up phase for the given model by freezing and unfreezing specific layers 
    in a stepwise manner. This function ensures that pre-trained layers are utilized effectively 
    while allowing new layers to adapt to the task.

    Args:
        A_to_B_model (torch.nn.Module): The model to be warmed up.
        filtered_dict (dict): Dictionary of pre-trained weights that match the model's layers.
        not_matching_keys (list): List of keys that did not match during weight loading.

    Returns:
        torch.nn.Module: The warmed-up model with updated weights.

    Notes:
        - The warm-up process is divided into two phases:
            1. Freeze pre-trained layers and train only new layers.
            2. Unfreeze decoder layers and train them along with the new layers.
        - The number of epochs for each phase is determined by `args.freeze_epochs`.
        - The `freeze_layers` function is used to freeze pre-trained layers.
        - The `warm_up_model` function is used to train the model during each phase.
    """
    # Phase 1: Freeze pre-trained layers and train only new layers
    for name, param in A_to_B_model.named_parameters():
        param.requires_grad = True  # Ensure all layers are initially learnable
    A_to_B_model = freeze_layers(A_to_B_model, filtered_dict, not_matching_keys)
    A_to_B_model = warm_up_model(A_to_B_model, freeze_epochs=args.freeze_epochs // 2)

    # Phase 2: Unfreeze decoder layers and train them along with new layers
    for name, param in A_to_B_model.named_parameters():
        if "decoder" in name:  # Unfreeze decoder layers
            param.requires_grad = True
    A_to_B_model = warm_up_model(A_to_B_model, freeze_epochs=args.freeze_epochs)

    # Phase 3: Unfreeze all layers
    for name, param in A_to_B_model.named_parameters():
        param.requires_grad = True  # Ensure all layers are initially learnable

    return A_to_B_model

def check_pretained(A_to_B_model, filtered_dict, not_matching_keys, args):
        """
        Handles the initialization and warmup of a model based on pre-trained weights, 
        freezing layers, and resuming training conditions.

        Args:
            A_to_B_model: The model to be initialized or warmed up.
            filtered_dict: Dictionary containing the pre-trained weights.
            not_matching_keys: Keys that do not match between the model and pre-trained weights.

        Raises:
            ValueError: If pre-trained weights are being loaded for unsupported model types.

        Warnings:
            - If pre-trained weights are not loaded but freezing layers is attempted.
            - If pre-trained weights are loaded without freezing layers.
            - If resuming training, freezing layers will have no effect.
        """
        if args.load_pretrained and args.freeze_epochs!=0 and args.resume==None and args.network in ["SwinUNETR", "SwinUNETR_vit", "nnUNet"]:
            # Warmup model
            # first the layers without pre-trained
            # Then the decoder part only
            A_to_B_model = warmup_step(A_to_B_model, filtered_dict, not_matching_keys)
        elif args.load_pretrained and args.freeze_epochs!=0 and args.resume==None and args.network not in ["SwinUNETR", "SwinUNETR_vit", "nnUNet"]:
            raise ValueError("ERROR: You are trying to load pre-trained weights, but the model is not SwinUNETR_vit, SwinUNETR, or nnUNet. The pre-trained weights cannot be loaded. Please ensure this is intentional.")
        elif not args.load_pretrained and args.freeze_epochs!=0 and args.resume==None:
            print("WARNING: You are not loading pre-trained weights, so freezing layers will have NO effect. NO WARMUP. Please ensure this is intentional.")
        elif args.load_pretrained and args.freeze_epochs==0 and args.resume==None:
            print("WARNING: You are loading pre-trained weights, but not freezing any layers. NO WARMUP. Please ensure this is intentional. Recommended 10-20 epochs of freezing.")
        elif args.resume!=None:
            print("WARNING: You are resuming training, so freezing layers will have NO effect. NO WARMUP. Please ensure this is intentional.")
        
def get_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--network', type=str, default='SwinVIT', help='Network type: SwinVIT (default), Unet, SwinUNETR_vit (for pre-trained vit), SwinUNETR (scratch or pre-trained SwinUNETR), nnUNet (scratch or pre-trained TotalSegmentator)')
    parser.add_argument('--batch_size_train', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 64], help='Patch size in (x, y, z)')
    parser.add_argument('--patch_num', type=int, default=2, help='Number of patches extracted per volume')
    parser.add_argument('--dataset_path', type=str, default='../../Dataset', help='Path to the dataset')
    parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate for MONAI DataLoader CacheDataset')
    parser.add_argument('--eval_metric', type=str, choices=['L1', 'L2'], default='L1', help='Evaluation metric choice: L1 or L2')
    parser.add_argument('--train_metric', type=str, choices=['MAE', 'MSE'], default='MSE', help='Train metric choice: MAE (L1) or MSE (L2)')
    parser.add_argument('--task', type=str, choices=['Task1', 'Task2'], default='Task1', help='Task selection: Task1 or Task2')
    parser.add_argument('--timestep_respacing', type=str, help='Timestep respacing values, e.g., 50 100')
    parser.add_argument('--timestep_respacing_val', type=str, help='Timestep respacing values for validation, e.g., ddim50')
    parser.add_argument('--verbose', action='store_true', help='Verbose output for detailed information')
    parser.add_argument('--shuffle', action='store_true', help='Use shuffle in the data loader')
    parser.add_argument('--sw_batch_size', type=int, default=12, help='Sliding window batch size')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap for sliding window')
    parser.add_argument('--overlap_mode', type=str, default='constant', help='Overlap mode for sliding window. constant or gaussian')
    parser.add_argument('--path_checkpoint', type=str, default='../../results/', help='Path to save model checkpoints')
    parser.add_argument('--load_pretrained', action='store_true', help='Load pre-trained weights.')
    parser.add_argument('--path_pretrained', type=str, help='Path to the pre-trained weights')
    parser.add_argument('--freeze_epochs', type=int, default=0, help='For how many epochs the loaded weights should be frozen')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from, e.g., args.path_checkpoint/A_to_B_ViTRes1_latest.pt')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--pacience', type=int, default=10, help='Number of epochs to wait before stoppin the training if val does not improve.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout to apply to the model.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to start with.')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval (in epochs)')
    parser.add_argument('--region', nargs='+', help='List of regions (head-and-neck, thorax and abdomen). To select all regions give: HN TH AB')
    parser.add_argument('--region_clip', action='store_true', help='Clip CT HU intensities by region. Only effective when doing region based training. clip_min_ct and clip_max_ct will be ignored if True')
    parser.add_argument('--clip_min_ct', type=int, default=-1000, help='Min value to clip CT HU values')
    parser.add_argument('--clip_max_ct', type=int, default=2000, help='Max value to clip CT HU values')
    parser.add_argument('--data_norm_mri', type=str, default='ScaleIntensityRanged', help='Data normalisation type. ScaleIntensityRanged (Default) or NormalizeIntensityd.')
    parser.add_argument('--data_norm_ct', type=str, default='ScaleIntensityRanged', help='Data normalisation type. ScaleIntensityRanged (Default) or NormalizeIntensityd.')
    parser.add_argument('--mri_clip_percentile', action='store_true', help='Clip MRI intensities 0.1, 99.9. Only afect during inference. Training has this by default.')
    parser.add_argument('--prob', type=float, default=0, help='Probability of applying transforms (data augmentation aggressiveness)')
    parser.add_argument('--add_train_metric', nargs='+', type=str, default=[], help='List of metrics to compute loss. Default is an empty list.' )
    parser.add_argument('--add_train_metric_weight', nargs='+', type=float, default=[], help='List of weights for each add_train_metric metric. Default is an empty list.' )
    parser.add_argument('--seg_path_weights', type=str, default="/projects/nian/synthrad2025/src/metrics/evaluation/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth", help='Path to the pre-trained totalSegmnetator weights')
    parser.add_argument('--noise_schedule', type=str, default="linear", help='Noise scheduler. linear (default) or cosine')
    parser.add_argument('--use_cosine_scheduler', action='store_true', help='If use cosine learning rate scheduler. it reduces from args.lr down to 1e-6.')
    parser.add_argument('--penalize_high_variance', action='store_true', help='Penalize the model if the variance infered is too high. This is used to avoid Nan in lower sampling steps.')
    parser.add_argument('--random_T_steps', action='store_true', help='Train the model to deal with several distinct T steps. Good for variable sampling later. Activate penalize_high_variance for better performance.')
    parser.add_argument('--all_training_data', action='store_true', help='In case of using training and validation sets for training.')
    parser.add_argument('--finetune', action='store_true', help='In case of finetuning. It will require resume arg to exist and will reset the lr scheduler.')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_value = 42
    set_complete_seed(seed_value)
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    name = f"VS-DDPM_{args.task}_{args.batch_size_train}_{args.n_epochs}_timestep_{args.timestep_respacing}_patchsize_{args.patch_num}_{args.network}_{args.train_metric}"
    for add_train in args.add_train_metric:
        name += "_"+str(add_train)
    if args.load_pretrained:
        name += "_pretrained"
    for patch_size_idx in args.patch_size:
        name += "_"+str(patch_size_idx)

    if args.task=='Task1':
        if args.data_norm_ct!=args.data_norm_mri:
            name += f"_distinctNorm"
        else:
            name += f"_{args.data_norm_ct}"
    elif args.task=='Task2':
        name += f"_{args.data_norm_ct}"

    name +=  "_region"
    for region_name in args.region:
        name += "_"+str(region_name)
    
    # ADD noise scheduler name 
    name += f"_{args.noise_schedule}"
    
    if args.prob != 0:
        name += f"_DA_{args.prob}"

    if args.penalize_high_variance:
        name += f"_pen_var"
    if args.random_T_steps:
        name += f"_random_T"

    name += "_CTminmax"
    name += str(args.clip_min_ct)
    name += "_"
    name += str(args.clip_max_ct)

    if args.all_training_data:
        name += f"_allTrainingData"

    if args.finetune:
        name += f"_finetune"

    args.path_checkpoint = join(args.path_checkpoint, name)
    wandb.init(project="SynthRad2025", name=name, dir=args.path_checkpoint)
    args.path_checkpoint = wandb.run.dir

    # Create checkpoint folder
    os.makedirs(f"{args.path_checkpoint}/scans",exist_ok=True)
    os.makedirs(f"{args.path_checkpoint}/model",exist_ok=True)

    if args.verbose:
        print("########### Arguments used ###########")
        print(f"Batch Size (Train): {args.batch_size_train}")
        print(f"Number of Workers: {args.num_workers}")
        print(f"Patch Size: {args.patch_size}")
        print(f"patch_num Size: {args.patch_num}")
        print(f"Dataset Path: {args.dataset_path}")
        print(f"Cache Rate: {args.cache_rate}")
        print(f"Task: {args.task}")
        print(f"Timestep Respacing: {args.timestep_respacing}")
        print(f"Timestep Respacing Validation: {args.timestep_respacing_val}")
        print(f"Sliding Window Batch Size: {args.sw_batch_size}")
        print(f"Overlap: {args.overlap}")
        print(f"Overlap mode: {args.overlap_mode}")
        print(f"Checkpoint Path: {args.path_checkpoint}")
        print(f"Resume Training: {args.resume}")
        print(f"Number of Epochs: {args.n_epochs}")
        print(f"Validation Interval: {args.val_interval} epochs")
        print(f"Regions: {args.region}")
        print(f"Do region clip: {args.region_clip}")
        print(f"Clip min CT HU: {args.clip_min_ct}")
        print(f"Clip max CT HU: {args.clip_max_ct}")
        print(f"Data normalisation function CT: {args.data_norm_ct}")
        print(f"Data normalisation function MRI: {args.data_norm_mri}")
        print(f"Using add_train_metric: {args.add_train_metric}")
        print(f"Using add_train_metric_weight: {args.add_train_metric_weight}")
        print(f"Using seg_path_weights: {args.seg_path_weights}")
        print(f"Using dropout: {args.dropout}")
        print(f"Using noise schedule: {args.noise_schedule}")
        print(f"penalize_high_variance: {args.penalize_high_variance}")
        print(f"random_T_steps: {args.random_T_steps}")
        print(f"Using device: {device}")
        print(f"Using all_training_data: {args.all_training_data}")
        print(f"use_cosine_scheduler: {args.use_cosine_scheduler}")
        print(f"Using learning rate: {args.lr}")
        
        print("####################################")

        if args.finetune:
            if args.prob != 0:
                args.prob = 0
                warnings.warn(
                    "Data augmentation probability set to 0 for fine-tuning to avoid artefacts.",
                    UserWarning
                )
            if args.train_metric != 'MAE':
                args.train_metric = 'MAE'
                warnings.warn(
                    "MSE should not be used for fine-tuning. Changing to MAE to make sharper outputs.",
                    UserWarning
                )
                if 'MAE' in args.add_train_metric:
                    args.add_train_metric.remove('MAE')
                    warnings.warn(
                        f"Removing duplicated MAE loss. Using: {args.add_train_metric}",
                        UserWarning
                    )
        
    ########################################
    ######### Define data loader ##########
    ########################################
    data_list_task_train, data_list_task_val = get_data_list(
        dataset_path=args.dataset_path, 
        task_datasplit_json=join(args.dataset_path, f"{args.task}_data_split.json"), # Assumes datasplit to be in the root folder of Dataset 
        task=args.task, 
        region=args.region,
        args=args
        )

    if args.all_training_data:
        print(f"Using all data for training. Validation results biased!")
        print(f"Total number of cases for training and validation before joining: {len(data_list_task_train)} : {len(data_list_task_val)}")
        data_list_task_train = data_list_task_train + data_list_task_val
        print(f"Total number of cases for training and validation before joining: {len(data_list_task_train)} : {len(data_list_task_val)}")

    print(f"{len(data_list_task_train)} cases for training and {len(data_list_task_val)} for validation.")
    train_dataloader, val_dataloader, train_transforms, val_transforms = get_dataloader(
        data_list_task_train=data_list_task_train,
        data_list_task_val=data_list_task_val,
        task=args.task,
        args=args,
        )

    ##################################################################
    ############ Loss functions and validation metrics ###############
    ##################################################################
   
    # For the evaluation step (NormalizeIntensityd has a variable range)
    if args.data_norm_ct == 'NormalizeIntensityd':
        args.clip_denoised = False
    elif args.data_norm_ct == 'ScaleIntensityRanged':
        args.clip_denoised = True
    
    # Used for training
    args.more_train_metric = {}
    add_train_metric = list(args.add_train_metric)

    if args.data_norm_ct == 'ScaleIntensityRanged': # -1 1
        data_range = 2
        args.max_clip = 1
        args.min_clip = -1
    elif args.data_norm_ct == 'NormalizeIntensityd': # z-score
        normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{args.task}_Train_normalization_stats_{args.clip_min_ct}_{args.clip_max_ct}.json"
        with open(normalization_stats_path, "r") as stats_file:
            normalization_stats = json.load(stats_file)
        ct_mean = normalization_stats.get("ct_mean")
        ct_std = normalization_stats.get("ct_std")
    
        min_z = (args.clip_min_ct - ct_mean) / ct_std
        max_z = (args.clip_max_ct - ct_mean) / ct_std

        args.max_clip = max_z
        args.min_clip = min_z

        data_range = max_z - min_z
    else:
        raise ValueError("'data_norm_ct' should be either ScaleIntensityRanged or NormalizeIntensityd")
        
    for add_metric in args.add_train_metric:
        ### Adding he PSNR
        if add_metric=='PSNR':
            print(f"Using PSNR as loss function. NOT RECOMMENDED!")
            psnr_func = PSNRMetric(
                    max_val=data_range, 
                    reduction='mean', 
                    get_not_nans=False)

            args.more_train_metric['PSNR'] = psnr_func
            add_train_metric.remove('PSNR')
        ### Adding the MAE loss function
        if add_metric=='MAE':
            print(f"Using MAE as loss function")
            args.more_train_metric['MAE'] =  torch.nn.L1Loss()
            add_train_metric.remove('MAE')
        ### Adding the SSIM loss function
        if add_metric=='SSIM':
            print(f"Using SSIM as loss function")
            args.more_train_metric['SSIM'] = SSIMLoss(
                                                spatial_dims=3,
                                                data_range=2,
                                                kernel_type='gaussian',
                                                win_size=11,
                                                kernel_sigma=1.5,
                                                k1=0.01,
                                                k2=0.03,
                                                reduction='mean')
            add_train_metric.remove('SSIM')
        if add_metric=='SSIM_tahn':
            print(f"Using SSIM_tahn as loss function")
            args.more_train_metric['SSIM_tahn'] = SSIMLoss(
                                                spatial_dims=3,
                                                data_range=2,
                                                kernel_type='gaussian',
                                                win_size=11,
                                                kernel_sigma=1.5,
                                                k1=0.01,
                                                k2=0.03,
                                                reduction='mean')
            add_train_metric.remove('SSIM_tahn')
        ### Direct dice score for the segmented cases
        if add_metric=='DSC':
            args.more_train_metric['DSC'] = TotalSegmentatorLoss(
                                    ct_std=ct_std, 
                                    ct_mean=ct_mean, 
                                    clip_min_ct=args.clip_min_ct, 
                                    clip_max_ct=args.clip_max_ct,
                                    path_weights=args.seg_path_weights
                                )
            add_train_metric.remove('DSC')
        ### Perceptual loss AFP
        if add_metric=='AFP':
            AFP_loss = AFP(net="TotalSeg_ABTH_V2", mae_weight=0.0, layers=[0, 1, 2, 3, 4, 5, 6, 7, 8], data_norm_ct=args.data_norm_ct, clip_min_ct=args.clip_min_ct, clip_max_ct=args.clip_max_ct, patch_size=args.patch_size)
            args.more_train_metric['AFP'] = AFP_loss
            add_train_metric.remove('AFP')
    print(f"Metrics used for training: {args.train_metric} and {args.more_train_metric}. Not used: {add_train_metric}")

     # Used for evaluation 
    eval_MAE = torch.nn.L1Loss()
    eval_MSE = torch.nn.MSELoss()
    eval_SSIM = SSIMLoss(
                spatial_dims=3,
                data_range=data_range,
                kernel_type='gaussian',
                win_size=11,
                kernel_sigma=1.5,
                k1=0.01,
                k2=0.03,
                reduction='mean'
            )
    eval_PSNR = PSNRMetric(
        max_val=args.clip_max_ct - args.clip_min_ct, 
        reduction="mean", 
        get_not_nans=False)
    """
    eval_SSIM = SSIMMetric(
        spatial_dims=3, 
        data_range=1.0, 
        kernel_type="gaussian", 
        win_size=11, 
        kernel_sigma=1.5, 
        k1=0.01, 
        k2=0.03, 
        reduction="mean", 
        get_not_nans=False
        )
    """

    ########################################
    ###### Build the MC-IDDPM process ######
    ########################################
    # The MC-IDDPM process is a combination of the DDPM and the transformer network. 
    # The DDPM is used to denoise the input image
    
    # In case we want to use always the same T steps
    #train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
    #    timestep_respacing=args.timestep_respacing,
    #    timestep_respacing_val=args.timestep_respacing_val,
    #    args=args
    #    )
        
    A_to_B_model, filtered_dict, not_matching_keys = get_model(
        device=device,
        args=args
        )
    # Check if the pre-trained weights are loaded correctly if they are required
    check_pretained(A_to_B_model, filtered_dict, not_matching_keys, args)
    
    # Print the number of parameters in the model
    if args.verbose:
        print('parameter number is '+str(sum(p.numel() for p in A_to_B_model.parameters())))
    
    # optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5, weight_decay=1e-4) # Old!
    optimizer = torch.optim.AdamW(
            A_to_B_model.parameters(),  # frozen encoder params will be skipped
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

    scaler = torch.amp.GradScaler("cuda")
    
    ########################################
    ############## Inference ###############
    ########################################
    # Use the sliding window  method to translate the whole Input image to CT volume. Must used it.
    # For example, if your whole volume is 64x64x64, and our window size is 64x64x4, so the function will automatically sliding down
    # the whole volume with a certain overlapping ratio
    # The window size (args.patch_size) is shown in the "Build the data loader using the monai library" section.
    # args.patch_size: the size of sliding window
    # img_num: the number of sliding window in each process, only related to your gpu memory, it will still run through the whole volume
    # overlap: the overlapping ratio
    inferer = SlidingWindowInferer(
        roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), 
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap, 
        mode=args.overlap_mode, 
        progress=True
        )

    # Smoother to avoid noisy validation early stop
    smoother = EMASmoother(alpha=0.99)
    ########################################
    ############ Training Loop #############
    ########################################
    if args.resume!=None:
        checkpoint = torch.load(join(args.resume), weights_only=False)
        A_to_B_model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['best_loss'] 
        if args.finetune and 'finetune' not in args.resume: 
            begin_epoch = 0
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
            # Retrieve the epoch and best loss
            begin_epoch = checkpoint['epoch']
        
        #best_psnr = checkpoint['best_psnr'] 
        ### TODO remove
        #best_loss = 10000 # TODO remove
        best_psnr = 0 # TODO remove
        #begin_epoch = 0 # TODO remove
        print(f"Loaded from: {args.resume}")
    else:
        if args.finetune:
            raise ValueError("--resume cannot be empty when trying to finetune a model!") 
        best_loss = 10000
        best_psnr = 0
        begin_epoch = 0
        print("Start training from scratch")
            
    if args.use_cosine_scheduler:
        cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(args.n_epochs - begin_epoch),
                eta_min=1e-6
            )

    pacience_counter = 0

    for epoch in range(begin_epoch, args.n_epochs): 
        print('Epoch:', epoch)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}") # TODO remove
        start_time = time.time() 

        if args.random_T_steps:
            possible_T = [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
            args.timestep_respacing = random.choice(possible_T)
   
        train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
            timestep_respacing=str(args.timestep_respacing),
            timestep_respacing_val=args.timestep_respacing_val,
            args=args
            )

        # Build the training function. Run the training function once = one epoch
        A_to_B_losses, A_to_B_model, optimizer, scaler = train(
            model=A_to_B_model, 
            optimizer=optimizer, 
            data_loader1=train_dataloader,
            scaler=scaler,
            for_totalsegmentator=("DSC" in args.add_train_metric),
            args=args
            )
        if args.use_cosine_scheduler:
            cosine_scheduler.step()

        wandb.log({
            "epoch": epoch, # Log the actual epoch number to the 'epoch' metric
            "lr": optimizer.param_groups[0]['lr'] # Use optimizer here, not opt_ae if it's passed as arg
            })

        average_loss_train = 0
        for idx, new_metric in enumerate(args.add_train_metric):
            wandb.log({f"Epoch_{new_metric}": np.nanmean(A_to_B_losses[new_metric]), "epoch": epoch})
            average_loss_train += np.nanmean(A_to_B_losses[new_metric])
            

        wandb.log({f"Epoch_{args.train_metric}": np.nanmean(A_to_B_losses['loss']), "epoch": epoch})
        average_loss_train += np.nanmean(A_to_B_losses['loss'])

        wandb.log({"Loss epoch": average_loss_train.item(), "epoch": epoch}) # wandb save
        print('Averaged loss is: '+ str(average_loss_train))
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        
        # Validation every epoch
        print("Evaluating...")
        loss_val = evaluate(
            model=A_to_B_model, 
            epoch=epoch, 
            data_loader1=val_dataloader,
            inferer=inferer,
            args=args
            )
        try:
            maes = [d['MAE'].mean().item() for d in loss_val]
            mean_mae = np.mean(maes)
        except:
            print(f"Error Computing the mean MAE for validation")
            mean_mae = 1

        try:
            mses = [d['MSE'].mean().item() for d in loss_val]
            mean_mse = np.mean(mses)
        except:
            print(f"Error Computing the mean MSE for validation")
            mean_mse = 1

        try:
            psnrs = [d['PSNR'].mean().item() for d in loss_val]
            mean_psnr = np.mean(psnrs)
        except:
            print(f"Error Computing the mean PSNR for validation")
            mean_psnr = 0

        try:
            ssims = [d['SSIM'].mean().item() for d in loss_val]
            mean_ssim = np.mean(ssims)
        except:
            print(f"Error Computing the mean SSIM for validation")
            mean_ssim = 0

        wandb.log({f"Val MAE": mean_mae, "epoch": epoch}) # wandb save
        wandb.log({f"Val MSE": mean_mse, "epoch": epoch}) # wandb save
        wandb.log({f"Val PSNR": mean_psnr, "epoch": epoch}) # wandb save
        wandb.log({f"Val SSIM": mean_ssim, "epoch": epoch}) # wandb save

        #mean_val_losses = ((mean_mae) + (mean_mse) + (1-mean_ssim))/3 # Here the PSNR is ignored since it's not normalised 
        # — EMA smoothing —
        smoothed_val_losses = smoother.update(mean_mae)
        # Save model with best validation loss!
        if smoothed_val_losses < best_loss:
            print(f'New validation best 🎉 Save the latest best model. smoothed_val_mse={smoothed_val_losses} | Real mean_mae={mean_mae}')
            pacience_counter = 0
            checkpoint = {
                'model_state_dict': A_to_B_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'best_psnr': mean_psnr
            }
            # Save the checkpoint dictionary
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_best.pt'))
            best_loss = smoothed_val_losses
        else:
            print(f"Epoch smooth validation loss: {smoothed_val_losses} | Epoch validation mean_mae: {mean_mae} | Best loss: {best_loss}")
            if epoch>(args.n_epochs//3)*2: # Only start counting pacience after 2/3 of the epochs ran
                pacience_counter += 1
                print(f"Pacience increased by 1. Value {pacience_counter} out of {args.pacience} epochs. Epoch smooth validation loss: {smoothed_val_losses} | Epoch validation mean_mae: {mean_mae} | Best loss: {best_loss}")

        # Save model with best PSNR
        if mean_psnr > best_psnr:
            print('Save the latest best model psnr')
            checkpoint = {
                'model_state_dict': A_to_B_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'best_psnr': mean_psnr
            }
            # Save the checkpoint dictionary
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_best_psnr.pt'))
            best_psnr = mean_psnr
        
        # Save every 10 epochs 
        if (epoch+1) % 10 == 0: 
            checkpoint = {
                    'model_state_dict': A_to_B_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': average_loss_train.item()
                }
            torch.save(checkpoint, join(args.path_checkpoint, "model", 'A_to_B_model_latest.pt')) 
        
        # save mid checkpoints
        if (epoch+1) % 100 == 0: 
            checkpoint = {
                    'model_state_dict': A_to_B_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_loss': average_loss_train.item()
                }
            torch.save(checkpoint, join(args.path_checkpoint, "model", f'A_to_B_model_{epoch}.pt')) 
        
        if pacience_counter >= args.pacience:
            print(f"The validation results are not improving for {args.pacience} epochs!")
            print(f"Stopping training")
            break
    wandb.finish()
    
