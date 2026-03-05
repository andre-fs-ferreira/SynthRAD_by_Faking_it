print("Start importing...")
import os
import argparse
import random
import monai
import json
import sys
sys.path.append("..")
sys.path.append(".")
from train_mc_IDDPM import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from os.path import join
from os import listdir
from os.path import isdir
from pathlib import Path
from itertools import islice
import time
import torch
from diffusion.Create_diffusion import *
from diffusion.resampler import *
import torch.nn.functional as F
from monai.transforms import (
    CropForeground
)
#from MonaiDataLoader import MonaiDataLoader
import nibabel as nib
from tqdm.auto import tqdm
from network.Diffusion_model_transformer import *
from network.Pre_trained_networks import load_pretrained_swinvit, load_pretrained_SwinUNETR, load_pretrained_TotalSegmentator, freeze_layers
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, Dataset, ITKReader
from monai.data.utils import dense_patch_slices
from monai.inferers.utils import _get_scan_interval
import SimpleITK as sitk
from typing import Callable, Tuple
import numpy as np
from tqdm.auto import tqdm
from itertools import product
print("Finished importing.")


##############################################################
##### Functions to prepare the input and post processing #####
##############################################################
def pad_if_smaller_symmetric(in_tensor, patch_size):
    """
    Pads tensor symmetrically if D/H/W is smaller than the corresponding patch size.
    Assumes input tensor shape: (B, C, D, H, W)
    Returns padded tensor and the padding applied.
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
    pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2 = padding

    if pad_d1 + pad_d2 > 0:
        in_tensor = in_tensor[:, :, pad_d1:-pad_d2 if pad_d2 > 0 else None, :, :]
    if pad_h1 + pad_h2 > 0:
        in_tensor = in_tensor[:, :, :, pad_h1:-pad_h2 if pad_h2 > 0 else None, :]
    if pad_w1 + pad_w2 > 0:
        in_tensor = in_tensor[:, :, :, :, pad_w1:-pad_w2 if pad_w2 > 0 else None]
    return in_tensor

def invert_crop_foreground(mask, sampled_images, min_intensity):
    """
    Reverses the cropping operation by padding the sampled images back into their original positions 
    within the tensor shape defined by the mask's bounding box. Also converts back from RAS orientation to RAI.

    This function uses the bounding box computed from the mask to determine the region where the 
    sampled images should be placed. The rest of the tensor is filled with a default background value.

    Args:
        mask (torch.Tensor): A binary or multi-class mask tensor used to compute the bounding box 
                             for cropping. Shape: (B, C, D, H, W).
        sampled_images (torch.Tensor): The tensor containing the cropped predictions to be padded 
                                        back into the original tensor shape. Shape: (B, C, d, h, w).

    Returns:
        torch.Tensor: A tensor with the same shape as the mask, where the sampled images are placed 
                      back into their original positions, and the remaining regions are filled with 
                      a default background value.
    """
    # Initialize the CropForeground transform
    crop_foreground = CropForeground()

    # Compute the bounding box from the mask
    start, end = crop_foreground.compute_bounding_box(img=mask.numpy())
    #print(f"start: {start}")
    #print(f"end: {end}")
    #print(f"mask.numpy(): {mask.numpy().shape}")
    #print(f"sampled_images.shape: {sampled_images.shape}")

    # Create a tensor of zeros with the same shape as the mask
    padded = np.ones_like(mask.numpy())*min_intensity

    # Extract the bounding box coordinates
    x0, y0, z0 = start[1], start[2], start[3]
    x1, y1, z1 = end[1], end[2], end[3]

    # Place the sampled images back into the padded tensor
    padded[:, :, x0:x1, y0:y1, z0:z1] = sampled_images

    padded = np.flip(padded[0][0], axis=(0, 1)) 
    
    return padded

def save_tensor_as_mha(image_tensor, mask, in_mha_filename, out_mha_filename, max_value=2000, min_value=-1000):
    """
    Save a PyTorch tensor as a .mha file using metadata from another .mha file.

    Parameters:
        image_tensor (torch.Tensor): The image data to save.
        mask (torch.Tensor): The mask tensor used for cropping and padding.
        in_mha_filename (str): Path to the reference .mha file to copy metadata from.
        out_mha_filename (str): Path where the output .mha file will be saved.
        max_value (float): Max value for normalization.
        min_value (float): Min value for normalization.
    """
    # Read metadata from reference file
    original_image = sitk.ReadImage(in_mha_filename)
    original_array = sitk.GetArrayFromImage(original_image)

    # Squeeze & rescale to [min, max]
    data = image_tensor.detach().cpu().numpy()
    if data.ndim == 5:
        data = data[0,0]
    elif data.ndim == 4:
        data = data[0]
    if args.data_norm_ct == 'ScaleIntensityRanged':
        data = np.clip(data, -1.0, 1.0)
        data = (data + 1) / 2
        data = data * (max_value - min_value) + min_value
    elif args.data_norm_ct == 'NormalizeIntensityd':
        normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{args.task}_Train_normalization_stats_{args.clip_min_ct}_{args.clip_max_ct}.json"
        with open(normalization_stats_path, "r") as stats_file:
            normalization_stats = json.load(stats_file)
        ct_mean = normalization_stats.get("ct_mean")
        ct_std = normalization_stats.get("ct_std")
        data = np.clip(data, args.min_clip, args.max_clip)
        data = data * ct_std + ct_mean
    # Pad back to the original shape
    
    # Apply smoothing and bilateral filtering
    if args.post_process: # Improve intesity metrics, reduce DSC and HD95
        print(f"Doing post process")
        data = smoothing_np(data, variance=args.post_variance)
        data = bilateral_smoothing_np(data, domain_sigma=args.post_domain_sigma, range_sigma=args.post_range_sigma)

    data = invert_crop_foreground(
                mask=mask, 
                sampled_images=data,
                min_intensity=min_value)
    


    # Transpose input from (X, Y, Z) to (Z, Y, X)
    data = np.transpose(data, (2, 1, 0))

    # Check shape compatibility
    if data.shape != original_array.shape:
        raise ValueError(f"Shape mismatch: tensor shape {data.shape} != reference shape {original_array.shape}")
    
    # Convert to SimpleITK image
    new_image = sitk.GetImageFromArray(data)  # Ensure type consistency

    # Copy metadata (spacing, origin, direction)
    new_image.CopyInformation(original_image)

    # Write the image
    sitk.WriteImage(new_image, out_mha_filename)

###################################################
##### Functions to prepare the inference loop #####
###################################################
def load_model(args):
        A_to_B_model, filtered_dict, not_matching_keys = get_model(
                device=args.device,
                args=args
                )
        checkpoint = torch.load(join(args.resume), weights_only=False)
        A_to_B_model.load_state_dict(checkpoint['model_state_dict'])
        # Retrieve the epoch and best loss
        begin_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss'] 
        print(f"Loaded from: {args.resume}")
        return A_to_B_model, begin_epoch, best_loss

def get_test_dataloader(test_set_path, args=None):
    # Creating a dict with the cases 
    # Find all .mr.mha files
    mr_mha_files = list(Path(test_set_path).rglob('*mr.mha'))
    all_files = []
    for mha_file in mr_mha_files:
        patient_id = str(mha_file).split('/')[-2]
        all_files.append(
            {
                args.key_in: str(mha_file),
                'ct': str(mha_file), # Work around to avoid errors loading
                'mask': str(mha_file).replace(f'{args.key_in}.mha', 'mask.mha')
            }
        )
    
    print(f"First 10 cases:")
    print(f"{all_files[:10]}")
    print(f"Doing {len(all_files)} cases.")
    print("#######")
    print(f"Min value to clip the CT scan: {args.clip_min_ct}")
    print(f"Min value to clip the CT scan: {args.clip_max_ct}")
    print(f"Data normalization for MRI: {args.data_norm_mri}")
    print(f"Data normalization for CT: {args.data_norm_ct}")

    return all_files 

def get_dataloop(args):
        data_list_task_train, data_list_task_val = get_data_list(
                dataset_path=args.dataset_path, 
                task_datasplit_json=join(args.dataset_path, f"{args.task}_data_split.json"), # Assumes dataplit to be in the root folder of Dataset 
                task=args.task, 
                region=args.region,
                args=args
                )
        print(f"data_list_task_train: {data_list_task_train[0]}")
        if args.test_set!=None:
            print(f"Replaced data_list_task_val with the testing set.")
            data_list_task_val = get_test_dataloader(
                test_set_path=args.test_set, 
                load_keys=[args.key_in,'mask'], 
                args=args
                )

        train_dataloader, val_dataloader, train_transforms, val_transforms = get_dataloader(
                data_list_task_train=data_list_task_train,
                data_list_task_val=data_list_task_val,
                task=args.task,
                args=args
                )
        return val_dataloader

def get_diffusion_scheduler(args):
    train_diffusion, val_diffusion, schedule_sampler = get_diffusion(
            timestep_respacing=args.timestep_respacing,
            timestep_respacing_val=args.timestep_respacing_val,
            args=args,
            )

    print(f"args.sw_batch_size: {args.sw_batch_size}")

    if args.crop_windows_borders:
        print(f"Cropping border!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Changing to overlap_mode gaussian")
        args.overlap_mode = 'gaussian'
        
        
    inferer = SlidingWindowInferer(
        roi_size=(args.patch_size[0], args.patch_size[1], args.patch_size[2]), 
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap, 
        sigma_scale=args.sigma_scale,
        mode=args.overlap_mode, 
        progress=True
        )
    print(f"Using: args.overlap {args.overlap}")

    return val_diffusion, inferer

def count_sliding_windows(image_size, roi_size, overlap=0.25):
    """
    Count how many patches (inferences) SlidingWindowInferer would run.

    Args:
      image_size (Sequence[int]): spatial shape of your input (e.g. [D, H, W] or [H, W]).
      roi_size    (Sequence[int]): the patch size you want to use.
      overlap     (float or Sequence[float]): fraction(s) of overlap between 0 and 1.

    Returns:
      int: the exact number of windows (= inferences) that will be executed.
    """
    num_dims = len(image_size)
    # ensure overlap is a sequence of length num_dims
    if isinstance(overlap, (float, int)):
        overlap = [float(overlap)] * num_dims
    elif len(overlap) != num_dims:
        raise ValueError(f"overlap must be a float or a sequence of length {num_dims}")

    # 1. Compute scan intervals (the “stride” per axis)
    scan_interval = _get_scan_interval(
        image_size=image_size,
        roi_size=roi_size,
        num_spatial_dims=num_dims,
        overlap=overlap,
    )
    # 2. Enumerate all patch‐slice definitions
    all_slices = dense_patch_slices(image_size, roi_size, scan_interval)
    # 3. Return how many there are
    return len(all_slices)

def compute_overlap(image_size, roi_size, target_num):
    """
    Compute the smallest scalar overlap such that the number of sliding windows is >= target_num.

    Uses binary search to find the smallest overlap where count_sliding_windows >= target_num.

    Args:
      image_size (Sequence[int]): spatial shape of your input (e.g. [D, H, W] or [H, W]).
      roi_size    (Sequence[int]): the patch size you want to use.
      target_num  (int): the desired number of windows.

    Returns:
      float: the smallest overlap value where count >= target_num, or None if impossible.
    """
    min_count = count_sliding_windows(image_size, roi_size, overlap=0.5)
    max_count = count_sliding_windows(image_size, roi_size, overlap=0.75)

    print(f"min_count: {min_count}")
    print(f"max_count: {max_count}")

    if target_num < min_count:
        return 0.5  
    if target_num > max_count:
        return 0.75

    low = 0.50
    high = 0.75
    best_overlap = low
    best_count = max_count

    for overlap_here in range(int(low*100), int(high*100), int(0.01*100)):
        overlap_here = overlap_here/100
        n_windows = count_sliding_windows(image_size, roi_size, overlap=overlap_here)
        print(f"overlap_here: {overlap_here} | n_windows: {n_windows}")
        if n_windows >= target_num:
            return best_overlap
        else:
            best_overlap = overlap_here

    return best_overlap

def smoothing_np(volume_np: np.ndarray, variance: float) -> np.ndarray:
    """
    Apply SimpleITK DiscreteGaussian smoothing to a numpy 3D volume.
    Returns a numpy array of same shape.
    """
    img = sitk.GetImageFromArray(volume_np.astype(np.float32))
    sm = sitk.DiscreteGaussian(img, variance=variance)
    return sitk.GetArrayFromImage(sm)

def bilateral_smoothing_np(volume_np: np.ndarray, domain_sigma: float = 2.0, range_sigma: float = 50.0) -> np.ndarray:
    """
    Apply SimpleITK Bilateral smoothing to a numpy 3D volume.
    Returns a numpy array of same shape.
    """
    img = sitk.GetImageFromArray(volume_np.astype(np.float32))
    bil = sitk.Bilateral(img, domainSigma=domain_sigma, rangeSigma=range_sigma)
    return sitk.GetArrayFromImage(bil)

def predict_loop(val_dataloader, diffusion_sampling, A_to_B_model, exp_name, max_intensity, min_intensity):
    if args.data_norm_mri == 'NormalizeIntensityd_Scaled':
        exp_name += "_new_NormalizeIntensityd_Scaled"
    
    if 'ddim' in args.timestep_respacing_val:
        exp_name += "ddim"
        args.use_DDIM = True
    else:
        args.use_DDIM = False

    if args.adjust_step:
        exp_name += "_adjustStep"

    if args.mask_crop:
        exp_name += "_maskCrop"

    if args.mask_crop_per_step:
        exp_name += "_maskCropPStep"
    
    if args.test_set!=None:
         exp_name += "_TESTSET"
    
    if args.crop_windows_borders:
        exp_name += f"_crop_borders_{args.sigma_scale}"
    
    if args.adjust_overlap:
        exp_name += f"_adjust_overlap_"
    else:
        exp_name += f"overlap{args.overlap}"

    if args.post_process:
        exp_name += f"post{args.post_variance}_{args.post_domain_sigma}_{args.post_range_sigma}"

    exp_name += f"_T_{args.timestep_respacing_val}"
    
    
    
    exp_name += f"_{args.TOTAL_EPOCHS}"
    for i, batch in enumerate(val_dataloader):
        args.overlap = args.original_overlap
        if i>=int(args.start_case) and i<=int(args.finish_case):
            print(f"Doing case {i} up from {args.start_case} up to {args.finish_case}.")
            start_time = time.time() 
            with torch.no_grad():
                condition = batch[args.key_in].to(args.device) 
                target = batch['ct'].to(args.device) 
                mask = batch['mask'].to(args.device) 

                if args.adjust_step:
                    input_shape = condition.shape
                    number_of_windows = count_sliding_windows(
                        image_size=(input_shape[-3], input_shape[-2], input_shape[-1]), 
                        roi_size=args.patch_size, 
                        overlap=args.overlap)
                    print(f"number_of_windows: {number_of_windows}")
                    # 198 -> windows | 7 T steps in this test | 574 seconds took on the online platform | 900 seconds in total to run 
                    args.timestep_respacing_val = ((900*198*7) // (574*number_of_windows)) # TODO change in case of using lighet networks

                    if args.adjust_step_light:
                        args.timestep_respacing_val = args.timestep_respacing_val * 3
                        print(f"New timestep_respacing_val multiplied by 3: {args.timestep_respacing_val}")

                    if args.target_discrete_T:
                        print(f"Starting with overlap={args.overlap}")
                        T_values = [5, 10, 15, 20, 25, 35, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
                        print(f"Start args.timestep_respacing_val: {args.timestep_respacing_val}")
                        old_timestep_respacing_val = args.timestep_respacing_val      
                        args.timestep_respacing_val = max(v for v in T_values if v <= args.timestep_respacing_val)
                        print(f"Converted {old_timestep_respacing_val} steps to {args.timestep_respacing_val} steps.")
                        if args.adjust_overlap:
                            # Considering this difference, the overlap can be increased as well
                            difference_of_timestep_respacing_val = int(args.timestep_respacing_val) - int(old_timestep_respacing_val)
                            if difference_of_timestep_respacing_val != 0: # just in case there is no difference, skip this step
                                total_windows_possible = ((900*198*7) // (574*args.timestep_respacing_val))
                                print(f"Number of windows possible in total={total_windows_possible}")
                                args.overlap = compute_overlap(
                                    image_size=(input_shape[-3], input_shape[-2], input_shape[-1]), 
                                    roi_size=args.patch_size, 
                                    target_num=total_windows_possible
                                    )
                                args.overlap = int(args.overlap * 100) / 100
                                print(f"New overlap possible! {args.overlap}")
                                if args.overlap<0.5:
                                    args.overlap = 0.5
                            
                            

                    if args.use_DDIM:
                        print(f"Using DDIM")
                        args.timestep_respacing_val = str(int(args.timestep_respacing_val))
                        args.timestep_respacing_val = f"ddim{args.timestep_respacing_val}"
                    else:
                        args.timestep_respacing_val = str(int(args.timestep_respacing_val))
                    val_diffusion, inferer = get_diffusion_scheduler(args) # get new val_diffusion for each timestep
                    print(f"Possible number of steps per window: {args.timestep_respacing_val}")
                else:
                    val_diffusion, inferer = get_diffusion_scheduler(args) # get new val_diffusion for each timestep
                print(f"condition.max(): {condition.max()}")
                print(f"condition.min(): {condition.min()}")

                condition, condition_padding = pad_if_smaller_symmetric(condition, args.patch_size)
                target, target_padding = pad_if_smaller_symmetric(target, args.patch_size) # TODO uncoment 
                mask, mask_padding = pad_if_smaller_symmetric(mask, args.patch_size)
                in_file_path = batch[f'{args.key_in}_meta_dict']['filename_or_obj'][0]
                region = batch[f'{args.key_in}_meta_dict']['filename_or_obj'][0].split('/')[-3]
                patient_id = batch[f'{args.key_in}_meta_dict']['filename_or_obj'][0].split('/')[-2]

                mha_filename = f'{args.out_path}/{exp_name}/{args.overlap_mode}/{region}/{patient_id}/ct_pred.mha'

                print(f"Doing: {mha_filename}")
                if os.path.exists(mha_filename):
                    print(f"File {mha_filename} already exists. Skipping...")
                    if '1HNC002' in mha_filename:
                        pass
                    else:
                        continue
                # Prediction
                if args.mask_crop or args.mask_crop_per_step:
                    input_tensor = torch.cat([condition, mask], dim=1)
                else:
                    input_tensor = condition
                with torch.amp.autocast("cuda"):
                    sampled_images = inferer(
                    input_tensor,
                    diffusion_sampling,                    
                    A_to_B_model,
                    val_diffusion,
                    args,
                    )

                sampled_images = unpad_if_smaller_symmetric(sampled_images, condition_padding)
                
                os.makedirs(mha_filename.replace('/ct_pred.mha', ''), exist_ok=True)
                save_tensor_as_mha(
                    image_tensor=sampled_images,
                    mask=batch['mask_full_res'],
                    in_mha_filename=in_file_path, 
                    out_mha_filename=mha_filename, 
                    max_value=max_intensity, 
                    min_value=min_intensity)

            args.overlap = 0.5 # this needs to be added in the end of the cycle for the next case. if not, the next computation is wrong

            print(f"File saved at {mha_filename}")
            print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
            
def diffusion_sampling(input_tensor, model, val_diffusion, args):
    if args.mask_crop:
        condition = input_tensor[:, :-1, :, :, :]
        mask = input_tensor[:, -1:, :, :, :]
    else:
        condition = input_tensor
    if 'ddim' in args.timestep_respacing_val:
        if args.mask_crop_per_step:
            sample_function = val_diffusion.ddim_sample_loop_mask
        else:
            print(f"Using ddim_sample_loop")
            sample_function = val_diffusion.ddim_sample_loop
    else:
        if args.mask_crop_per_step:
            print(f"Using p_sample_loop_mask")
            sample_function = val_diffusion.p_sample_loop_mask
        else:
            sample_function = val_diffusion.p_sample_loop
    if args.mask_crop_per_step:
        sampled_images = sample_function(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3],condition.shape[4]),
            condition=condition,
            clip_denoised=args.clip_denoised,
            min_clip_value=args.min_clip
            )
    else:
        sampled_images = sample_function(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3],condition.shape[4]),
            condition=condition,
            clip_denoised=args.clip_denoised
            )

    # Check in sampled_images if there are any NaN values
    if torch.isnan(sampled_images).any():
        print("Warning: Sampled images contain NaN values.")
        # Return 0 instead of NaN
        sampled_images = torch.nan_to_num(sampled_images, nan=0.0)

    sampled_images = torch.clamp(sampled_images, min=args.min_clip, max=args.max_clip)
    if args.mask_crop:
        print("Crop using mask")
        sampled_images[mask == 0] = args.min_clip # We know that outside of the mask should be background
    return sampled_images

def get_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--network', type=str, default='SwinVIT', help='Network type: SwinVIT (default), SwinUNETR_vit (for pre-trained vit), SwinUNETR (scratch or pre-trained SwinUNETR), nnUNet (scratch or pre-trained TotalSegmentator)')
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
    parser.add_argument('--timestep_respacing_val', type=str, help='Timestep respacing values for validation, e.g., ddim50 or 50')
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
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--val_interval', type=int, default=50, help='Validation interval (in epochs)')
    parser.add_argument('--region', nargs='+', help='List of regions (head-and-neck, thorax and abdomen). To select all regions give: HN TH AB')
    parser.add_argument('--region_clip', action='store_true', help='Clip CT HU intensities by region. Only effective when doing region based training')
    parser.add_argument('--clip_min_ct', type=int, default=-1000, help='Min value to clip CT HU values')
    parser.add_argument('--clip_max_ct', type=int, default=2000, help='Max value to clip CT HU values')
    parser.add_argument('--data_norm_mri', type=str, default='ScaleIntensityRanged', help='Data normalisation type. ScaleIntensityRanged (Default) or NormalizeIntensityd.')
    parser.add_argument('--data_norm_ct', type=str, default='ScaleIntensityRanged', help='Data normalisation type. ScaleIntensityRanged (Default) or NormalizeIntensityd.')
    parser.add_argument('--mri_clip_percentile', action='store_true', help='Clip MRI intensities 0.1, 99.9. Only afect during inference. Training has this by default.')
    parser.add_argument('--exp_name', type=str, help='Name of the folder wih the experiment, e.g., MC-IDDPM_Task1_2_5000_timestep_1000_patchsize_32_SwinVIT_MAE_64_64_4_region')
    parser.add_argument('--prob', type=float, default=0.0, help='Probability of applying transforms (data augmentation aggressiveness)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout to apply to the model.')
    parser.add_argument('--wandb_id', type=str, help='The random name defined by wandb')
    parser.add_argument('--TOTAL_EPOCHS', type=str, help='Total number of epohcs. Sum of all trainings!')
    parser.add_argument('--noise_schedule', type=str, default="linear", help='Noise scheduler. cinear (default) or cosine')
    parser.add_argument('--adjust_step', action='store_true', help='Adjust the number of sampling steps to fit 15 min.')
    parser.add_argument('--adjust_step_light', action='store_true', help='Adjust the number of sampling steps to fit 15 min for lighter models.')
    parser.add_argument('--target_discrete_T', action='store_true', help='Adjust the number of sampling steps to the trained values.')
    parser.add_argument('--mask_crop', action='store_true', help='Everything outside the mask will be set to the background value.')
    parser.add_argument('--mask_crop_per_step', action='store_true', help='Everything outside the mask will be set to the background value at every step in inference.')
    parser.add_argument('--start_case', type=float, help='Case number to start infer.')
    parser.add_argument('--finish_case',  type=float, help='Case number to start infer. Set -1 to run until the end.')
    parser.add_argument('--test_set',  type=str, default=None, help='If set, will use the path provided for inference. Should have the same otganization as the test set')
    parser.add_argument('--crop_windows_borders', action='store_true', help='If crop the borders of slidding windows predictions to avoid padding artifacts.')
    parser.add_argument('--sigma_scale', type=float, default=0.125, help='Sigma scale for gaussian blending.')
    parser.add_argument('--adjust_overlap', action='store_true', help='If compute what overlap better suits the solution. It will be never smaller than the pre-defined.')
    parser.add_argument('--post_process', action='store_true', help='If apply post_process.')
    parser.add_argument('--post_variance', type=float, help='Post process arg.')
    parser.add_argument('--post_domain_sigma', type=float, help='Post process arg.')
    parser.add_argument('--post_range_sigma', type=float, help='Post process arg.')
    parser.add_argument('--out_path',  type=str, default="/projects/nian/synthrad2025/experiments/MC-IDDPM/IDDPM", help='Path to save the predictions')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    seed_value = 42
    set_complete_seed(seed_value)
    args = get_args()
    torch.backends.cudnn.benchmark = True
    args.original_overlap = args.overlap 
    #######################
    ###### Arguments ######
    #######################
    ## To change
    #args.region_clip = False # Important!
    #args.patch_size = (64, 64, 4)
    #args.sw_batch_size = 16
    #args.overlap = 0.5
    if args.region_clip: 
        exp_name_L = ["HN",
                    "TH",
                    "AB"]
        wandb_id_L = [
            "latest-run",
            "latest-run",
            "latest-run"]
        region_L = [
            ["HN"], 
            ["TH"],
            ["AB"]]
        max_intensity_L = [1700, 1400, 1400]
        min_intensity_L = [-1024, -1024 ,-1024] 
    else:
        #exp_name = "MC-IDDPM_Task1_2_5000_timestep_1000_patchsize_32_SwinVIT_MAE_64_64_4_region"
        exp_name = args.exp_name
        #for region_name in args.region:
        #    exp_name += f"_{region_name}"

        #exp_name += f"_{args.noise_schedule}"
        
        wandb_id = args.wandb_id#"latest-run"#"latest-run"
        #args.region = ["TH"]
        max_intensity = args.clip_max_ct
        min_intensity = args.clip_min_ct
        
    if args.task == "Task1":
        args.key_in = 'mr'
    elif args.task == "Task2":
        args.key_in = 'cbct'

    # Fairly constant
    key_out = 'ct'
    key_mask = 'mask'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #args.network = "SwinVIT"
    args.cache_rate = 0
    args.batch_size_train = 2
    args.eval_metric = "L1"
    args.train_metric = "MAE"
    args.patch_num = 1
    args.shuffle = False
    
    print("########### Arguments used ###########")
    print(f"Network: {args.network}")
    print(f"Total training epochs: {args.TOTAL_EPOCHS}")
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
    print(f"Using device: {args.device}")
    print(f"Using noise schedule: {args.noise_schedule}")
    print(f"adjust_step: {args.adjust_step}")
    print(f"Test set: {args.test_set}")
    print(f"args.sigma_scale: {args.sigma_scale}")
    print(f"args.post_process: {args.post_process}")
    print(f"args.post_variance: {args.post_variance}")
    print(f"args.post_domain_sigma: {args.post_domain_sigma}")
    print(f"args.post_range_sigma: {args.post_range_sigma}")
    print("#####################################")

    args.add_train_metric = []
    # To ensure the intensioties are not clipped incorrectly
    if args.data_norm_ct == 'NormalizeIntensityd':
        args.clip_denoised = False # TODO this could be removing important values. Test with and without clip!
        normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{args.task}_Train_normalization_stats_{args.clip_min_ct}_{args.clip_max_ct}.json"
        with open(normalization_stats_path, "r") as stats_file:
            normalization_stats = json.load(stats_file)
        ct_mean = normalization_stats.get("ct_mean")
        ct_std = normalization_stats.get("ct_std")
        args.min_clip = (args.clip_min_ct - ct_mean) / ct_std
        args.max_clip = (args.clip_max_ct - ct_mean) / ct_std
    elif args.data_norm_ct == 'ScaleIntensityRanged':
        args.clip_denoised = True
        args.min_clip = -1
        args.max_clip = +1                    
    print(f"args.min_clip: {args.min_clip}")
    print(f"args.max_clip: {args.max_clip}")

    # Inference loop
    if args.region_clip:
        print(f"Doing region_clip: {args.region_clip}") 
        for exp_idx, region_name in enumerate(exp_name_L):
            if region_name is None:
                continue
            exp_name = f"{args.exp_name}_{region_name}"
            wandb_id = wandb_id_L[exp_idx]
            args.region = region_L[exp_idx]
            max_intensity = max_intensity_L[exp_idx] 
            min_intensity = min_intensity_L[exp_idx] 
            args.path_checkpoint = f"/projects/nian/synthrad2025/results/MC-IDDPM/{exp_name}/wandb/{wandb_id}/files"
            print(f"args.path_checkpoint: {args.path_checkpoint}")
            args.resume = join(args.path_checkpoint, "model/A_to_B_model_latest.pt")

            print(f"Doing region: {args.region}") 

            val_dataloader = get_dataloop(args)
            A_to_B_model, begin_epoch, best_loss = load_model(args)

            if torch.cuda.device_count() > 1:
                print(f"Let's use {torch.cuda.device_count()} GPUs!")
                A_to_B_model = nn.DataParallel(A_to_B_model)
            A_to_B_model.to(args.device)
            A_to_B_model.eval()

            print('Epoch:', begin_epoch)
            print('best_loss:', best_loss)
            predict_loop(
                val_dataloader=val_dataloader,
                diffusion_sampling=diffusion_sampling,
                A_to_B_model=A_to_B_model, 
                exp_name=exp_name,
                max_intensity=max_intensity,
                min_intensity=min_intensity
            )
    else:
        args.path_checkpoint = f"/projects/nian/synthrad2025/results/MC-IDDPM/{exp_name}/wandb/{wandb_id}/files"
        if args.TOTAL_EPOCHS=="FINAL":
            args.resume = join(args.path_checkpoint, "model/A_to_B_model_latest.pt")
        else:
            args.resume = join(args.path_checkpoint, f"model/A_to_B_model_{args.TOTAL_EPOCHS}.pt")
        
        print(f"Doing region: {args.region}") 
        
        val_dataloader = get_dataloop(args)
        A_to_B_model, begin_epoch, best_loss = load_model(args)

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            A_to_B_model = nn.DataParallel(A_to_B_model)
        A_to_B_model.to('cuda')
        A_to_B_model.eval()

        for name, param in A_to_B_model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in parameter: {name}")

        print('Epoch:', begin_epoch)
        print('best_loss:', best_loss)
        predict_loop(
            val_dataloader=val_dataloader,
            diffusion_sampling=diffusion_sampling, 
            A_to_B_model=A_to_B_model, 
            exp_name=exp_name,
            max_intensity=max_intensity,
            min_intensity=min_intensity
        )
