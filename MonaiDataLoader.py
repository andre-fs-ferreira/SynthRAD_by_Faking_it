from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    CropForegroundd,
    RandSpatialCropSamplesd,
    DeleteItemsd,
    NormalizeIntensityd,
    Padd,
    CopyItemsd,
    Resized,
    ToTensord,
    ToTensor,
    RandAffined,
    RandGridDistortiond,
    RandBiasFieldd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    ClipIntensityPercentilesd,
)

from monai.data import CacheDataset, DataLoader, Dataset 
from monai.data.utils import pad_list_data_collate
from os import listdir
from os.path import join
from os.path import isdir
from tqdm.auto import tqdm
import numpy as np
import json
import os
import nibabel as nib

class MonaiDataLoader:
    def __init__(self, train_data_files, val_data_files, spatial_size=(96, 96, 96), patch_num=2, key_in="mri", key_out="ct", key_mask="mask",
                 cache_rate=0.0, batch_size=2, shuffle=True, num_workers=1, region=['HN', 'TH', 'AB'], region_clip=False, a_min_ct=-1000, a_max_ct=2000,
                 data_norm_ct='ScaleIntensityRanged', data_norm_mri='ScaleIntensityRanged', for_totalsegmentator=False, prob=0.0, mri_clip_percentile=True):
        """
        Args:
            train_data_files (list[dict]): List of dictionaries with file paths for each image modality.
            val_data_files (list[dict]): List of dictionaries with file paths for each image modality.
            spatial_size (tuple[int]): Target spatial dimensions (H, W, D) for padding/cropping.
            patch_num (int): Number of patches extracted per volume
            cache_rate (float): Rate of caching data in CacheDataset.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the dataset.
            key_mask (str): Mask key
            key_in (str): Modality key to start with
            key_out (str): Modality key to predict
        """
        self.train_data_files = train_data_files
        self.val_data_files = val_data_files
        self.spatial_size = spatial_size
        self.patch_num = patch_num
        self.cache_rate = cache_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.key_mask = key_mask
        self.key_in = key_in
        self.key_out = key_out
        self.region = region
        self.region_clip = region_clip
        self.data_norm_ct = data_norm_ct
        self.data_norm_mri = data_norm_mri
        self.a_min_ct = a_min_ct
        self.a_max_ct = a_max_ct
        self.for_totalsegmentator = for_totalsegmentator
        self.prob = prob
        self.mri_clip_percentile = mri_clip_percentile

        self.all_keys = [key_in, 'mask', key_out, 'seg', 'foreground_start_coord', 'foreground_end_coord']
    
    def get_train_transforms(self):
        """
        Returns:
            A MONAI Compose object with the desired transforms:
              - Load images
              - Ensure channel-first format
              - Pad images smaller than the desired spatial size
              - Randomly crop a patch of the target spatial size
              - Convert images to tensors
        """
        assert self.key_out=="ct", "key_out should be 'ct'"

        # Defining the background padding value
        if self.key_in=="cbct":
            # Both cbct and ct have -1000 background
            pad_minus_1000_keys = [self.key_in, self.key_out]
            task_name = "Task2"
        else:
            # Only ct should have -1000 as background
            pad_minus_1000_keys = [self.key_out]
            task_name = "Task1"
        """
        if self.key_in=="mri":
            # Both mri and mask have 0 background
            pad_0_keys = [self.key_in, self.key_mask]
        else:
            # if cbct is the input, only the mask should have 0 as background
            pad_0_keys = [self.key_mask]
        """
        # Load image, ensure RAS orientation and random crop of a volume with the desired shape
        # Define Keys to load 
        if not self.for_totalsegmentator:
            load_keys = [self.key_in, self.key_out, self.key_mask]
        else:
            load_keys = [self.key_in, self.key_out, self.key_mask, 'seg']
            
        train_transforms_list = [
            LoadImaged(keys=load_keys),
            EnsureChannelFirstd(keys=load_keys),
            Orientationd(keys=load_keys, axcodes='RAS'),
            CropForegroundd(keys=load_keys, source_key=self.key_mask, allow_smaller=False),  # Crop based on mask
            #DeleteItemsd(keys=[self.key_mask]),
        ]

        # If prob of transforms is !=0, apply augmentations
        if self.prob != 0:
            print(f"Using Data Augmentation!")
            aug_transforms = self.get_aug_transforms()
            train_transforms_list += aug_transforms

        # Scale intensity values to the range [-1, 1]
        # Based on https://www.sciencedirect.com/science/article/pii/S1936878X2300339X
        # Based on https://pmc.ncbi.nlm.nih.gov/articles/PMC7067667/
        if self.region_clip:
            a_min_ct = -1024
            if 'HN' in self.region:
                a_max_ct = 1700
            else:
                a_max_ct = 1400
            
        else:
            a_min_ct = self.a_min_ct
            a_max_ct = self.a_max_ct

        print(f"Real a_min_ct: {a_min_ct}")
        print(f"Real a_max_ct: {a_max_ct}")
        if self.data_norm_ct == 'ScaleIntensityRanged':
            print(f"Doing CT ScaleIntensityRanged")
            train_transforms_list.append(ScaleIntensityRanged( 
                                            keys=pad_minus_1000_keys,
                                            a_min=a_min_ct,
                                            a_max=a_max_ct,
                                            b_min=-1.0,
                                            b_max=1.0,
                                            clip=True,
                                            )
                                        )
        elif self.data_norm_ct == 'NormalizeIntensityd': 
            print(f"Doing CT NormalizeIntensityd")
            normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{task_name}_Train_normalization_stats_{a_min_ct}_{a_max_ct}.json"
            if not os.path.exists(normalization_stats_path):
                self.create_stats_file(self.train_data_files, a_min_ct, a_max_ct, normalization_stats_path)
            with open(normalization_stats_path, "r") as stats_file:
                normalization_stats = json.load(stats_file)
            ct_mean = normalization_stats.get("ct_mean")
            ct_std = normalization_stats.get("ct_std")
            print(f"ct_mean: {ct_mean}")
            print(f"ct_std: {ct_std}")
            if ct_mean is None or ct_std is None:
                raise ValueError("ct_mean or ct_std not found in the normalization stats file.")
            if self.key_out=="ct":
                train_transforms_list.append(ScaleIntensityRanged( 
                    keys=[self.key_out],
                    a_min=a_min_ct,
                    a_max=a_max_ct,
                    b_min=a_min_ct,
                    b_max=a_max_ct,
                    clip=True,
                    )
                )
                train_transforms_list.append(NormalizeIntensityd(
                    keys=[self.key_out],
                    subtrahend=ct_mean,
                    divisor=ct_std,
                    nonzero=False,
                    channel_wise=False
                    )
                ) 
        else:
            raise ValueError(f"Unsupported data normalization method: {self.data_norm_ct}. Available ScaleIntensityRanged (linear, default) or 'NormalizeIntensityd' (z-score)")
        if self.key_in=="mri":
            if self.data_norm_mri == 'ScaleIntensityRanged':
                print(f"Doing MRI ScaleIntensityRanged")
                train_transforms_list.append(ClipIntensityPercentilesd(
                                                keys=[self.key_in],
                                                lower=0.1,
                                                upper=99.9
                                                )
                )
                train_transforms_list.append(ScaleIntensityd(
                                                keys=[self.key_in],
                                                minv=-1.0,
                                                maxv=1.0,
                                                )
                                        )
            elif self.data_norm_mri == 'NormalizeIntensityd': 
            
                # Added percentile clip because the max value were being too big!
                train_transforms_list.append(ClipIntensityPercentilesd(
                                                keys=[self.key_in],
                                                lower=0.1,
                                                upper=99.9
                                                )
                )

                train_transforms_list.append(NormalizeIntensityd(
                                                keys=[self.key_in],
                                                subtrahend=None,
                                                divisor=None,
                                                nonzero=False, 
                                                channel_wise=False
                                                )
                    ) 
            elif self.data_norm_mri == 'NormalizeIntensityd_Scaled': 
            
                print(f"Doing MRI NormalizeIntensityd_Scaled")
                # Added percentile clip because the max value were being too big!
                if self.mri_clip_percentile:
                    print(f"Clip MRI Percentile")
                    train_transforms_list.append(ClipIntensityPercentilesd(
                                                    keys=[self.key_in],
                                                    lower=0.1,
                                                    upper=99.9
                                                    )
                    )
                train_transforms_list.append(NormalizeIntensityd(
                                                keys=[self.key_in],
                                                subtrahend=None,
                                                divisor=None,
                                                nonzero=False,
                                                channel_wise=False
                                                )
                ) 
                train_transforms_list.append(ScaleIntensityd(
                                                keys=[self.key_in],
                                                minv=-1.0,
                                                maxv=1.0,
                                                )
                                        )
            else:
                raise ValueError(f"Unsupported data normalization method: {self.data_norm_mri}. Available ScaleIntensityRanged (linear, default) or 'NormalizeIntensityd' (z-score) or NormalizeIntensityd_Scaled (z-score and linear to -1 and 1)")
            
        
        # Random crop patch and resize to the desired spatial size
        #train_transforms_list.append(RandSpatialCropd( # It crops one volume per case, less efficient than RandSpatialCropSamplesd
        #                                keys=[self.key_in, self.key_out], 
        #                                roi_size=self.spatial_size,
        #                                random_center=True)
        #                            )
        if not self.for_totalsegmentator:
            train_transforms_list.append(RandSpatialCropSamplesd(
                                            keys=load_keys,
                                            roi_size=self.spatial_size,
                                            num_samples=self.patch_num,
                                            random_size=False)
                                        )

            train_transforms_list.append(ResizeWithPadOrCropd(
                                            keys=load_keys,
                                            spatial_size=self.spatial_size,
                                            mode="minimum")
                                            #value=-1)
                                        )
        else:
            print(f"INSIDE DSC transform!")
            train_transforms_list.append(ResizeWithPadOrCropd(
                                            keys=load_keys,
                                            spatial_size=(336, 336, 128),
                                            mode=["minimum", "minimum", "minimum", "minimum"])
                                            #value=-1)
                                        )
            train_transforms_list.append(
                                            CopyItemsd(keys=["seg"], times=1, names=["seg_downsample"], allow_missing_keys=False)
                                        )
            train_transforms_list.append(
                                            Resized(keys=['seg_downsample'], spatial_size=(112, 112, 128), mode=['nearest'])
                                        )
        
        train_transforms_list.append(ToTensord(self.all_keys, allow_missing_keys=True))
        train_transforms = Compose(train_transforms_list)
        
        return train_transforms

    def get_aug_transforms(self):
        keys_list = [self.key_in, self.key_out]  
        mode_list = ["trilinear", "trilinear"]
        aug_transforms = [
            ###############################
            ###### DATA AUGMENTATION ######
            # Based on https://arxiv.org/pdf/2006.06676.pdf
            # rotate 3 degrees
            # translation (16,32,8)
            # scale_range (-0.1, 0.1) -> zoom!
            # shear_range (-0.1, 0.1)
            
            RandAffined(
                keys=keys_list,
                prob=self.prob,
                rotate_range=((-np.pi/60,np.pi/60),(-np.pi/60,np.pi/60),(-np.pi/60,np.pi/60)), # 3 degrees
                #translate_range=(16,16,4), 
                scale_range=((-0.03,0.03),(-0.03,0.03),(-0.03,0.03)),
                shear_range=None,#((-0.05,0.05),(-0.05,0.05),(-0.05,0.05)),
                padding_mode="border",
                mode=mode_list,
                ),
            # RandGridDistortiond (Elastic Deformation)
            RandGridDistortiond(keys=keys_list, prob=self.prob, num_cells=(5, 5, 5), distort_limit=(0.01, 0.01, 0.01), padding_mode="border", mode=mode_list),
        ]

        if keys_list[0]=='mri':
            # Bias field
            aug_transforms.append(RandBiasFieldd(keys=[keys_list[0]], degree=3, coeff_range=(0.0, 0.05), prob=self.prob))
            #### Intensity #### 
            # Simulate Rand Gamma Image
            aug_transforms.append(RandShiftIntensityd(keys=[keys_list[0]], prob=self.prob, offsets=(-0.1, 0.1)))
            aug_transforms.append(RandScaleIntensityd(keys=[keys_list[0]], prob=self.prob, factors=(-0.1, 0.1)))
            aug_transforms.append(RandAdjustContrastd(keys=[keys_list[0]], prob=self.prob, gamma=(0.9, 1.1)))
            # Blur
            aug_transforms.append(RandGaussianSmoothd(keys=[keys_list[0]], sigma_x=(0.1, 0.4), sigma_y=(0.1, 0.4), sigma_z=(0.1, 0.4), approx='erf', prob=self.prob, allow_missing_keys=False))
            #Noise gaussian
            aug_transforms.append(RandGaussianNoised(keys=[keys_list[0]], prob=self.prob/2, mean=0, std=0.01))
            ### FINISH DATA AUGMENTATION ##
            ###############################
        
    
        return aug_transforms

    def get_val_transforms(self):
        """
        Returns:
            A MONAI Compose object with the desired transforms:
              - Load images
              - Ensure channel-first format
              - Pad images smaller than the desired spatial size
              - Randomly crop a patch of the target spatial size
              - Convert images to tensors
        """
        assert self.key_out=="ct", "key_out should be 'ct'"

        # Defining the background padding value
        if self.key_in=="cbct":
            # Both cbct and ct have -1000 background
            pad_minus_1000_keys = [self.key_in, self.key_out]
            task_name = "Task2"
        else:
            # Only ct should have -1000 as background
            pad_minus_1000_keys = [self.key_out]
            task_name = "Task1"

        print("For final inference, don't forget to pad back the output to the original size (before cropforegroundd)")
        # Load image, ensure RAS orientation and random crop of a volume with the desired shape
        val_transforms_list = [
            LoadImaged(keys=[self.key_in, self.key_out, self.key_mask], image_only=False),
            EnsureChannelFirstd(keys=[self.key_in, self.key_out, self.key_mask]),
            Orientationd(keys=[self.key_in, self.key_out, self.key_mask], axcodes='RAS'),
            CopyItemsd(keys=[self.key_mask], times=1, names=[f"{self.key_mask}_full_res"], allow_missing_keys=False),
            CropForegroundd(keys=[self.key_in, self.key_out, self.key_mask], source_key=self.key_mask, allow_smaller=False),  # Crop based on mask
        ]
        
        if self.region_clip:
            a_min_ct = -1024
            if 'HN' in self.region:
                a_max_ct = 1700
            else:
                a_max_ct = 1400
            
        else:
            a_min_ct = self.a_min_ct
            a_max_ct = self.a_max_ct

        # Scale intensity values to the range [-1, 1]
        # Based on https://www.sciencedirect.com/science/article/pii/S1936878X2300339X
        # Based on https://pmc.ncbi.nlm.nih.gov/articles/PMC7067667/
        if self.data_norm_ct == 'ScaleIntensityRanged':
            print(f"Doing CT ScaleIntensityRanged")
            val_transforms_list.append(ScaleIntensityRanged( 
                                            keys=pad_minus_1000_keys,
                                            a_min=a_min_ct,
                                            a_max=a_max_ct,
                                            b_min=-1.0,
                                            b_max=1.0,
                                            clip=True,
                                            )
            )
        elif self.data_norm_ct == 'NormalizeIntensityd': 
            print(f"Doing CT NormalizeIntensityd")
            normalization_stats_path = f"/projects/nian/synthrad2025/Dataset/{task_name}_Train_normalization_stats_{a_min_ct}_{a_max_ct}.json"
            if not os.path.exists(normalization_stats_path):
                self.create_stats_file(self.train_data_files, a_min_ct, a_max_ct, normalization_stats_path)
            with open(normalization_stats_path, "r") as stats_file:
                normalization_stats = json.load(stats_file)
            ct_mean = normalization_stats.get("ct_mean")
            ct_std = normalization_stats.get("ct_std")
            print(f"ct_mean: {ct_mean}")
            print(f"ct_std: {ct_std}")
            if ct_mean is None or ct_std is None:
                raise ValueError("ct_mean or ct_std not found in the normalization stats file.")
            if self.key_out=="ct":
                val_transforms_list.append(ScaleIntensityRanged( 
                    keys=[self.key_out],
                    a_min=a_min_ct,
                    a_max=a_max_ct,
                    b_min=a_min_ct,
                    b_max=a_max_ct,
                    clip=True,
                    )
                )
                val_transforms_list.append(NormalizeIntensityd(
                    keys=[self.key_out],
                    subtrahend=ct_mean,
                    divisor=ct_std,
                    nonzero=False,
                    channel_wise=False
                    )
                ) 
        else:
            raise ValueError(f"Unsupported data normalization method: {self.data_norm_ct}. Available ScaleIntensityRanged (linear, default) or 'NormalizeIntensityd' (z-score)")
        if self.key_in=="mri":
            if self.data_norm_mri == 'ScaleIntensityRanged':
                print(f"Doing MRI ScaleIntensityRanged")
                if self.mri_clip_percentile:
                    print(f"Clip MRI Percentile")
                    val_transforms_list.append(ClipIntensityPercentilesd(
                                                    keys=[self.key_in],
                                                    lower=0.1,
                                                    upper=99.9
                                                    )
                    )
                val_transforms_list.append(ScaleIntensityd(
                                                keys=[self.key_in],
                                                minv=-1.0,
                                                maxv=1.0,
                                                )
                                        )
            elif self.data_norm_mri == 'NormalizeIntensityd': 
                print(f"Doing MRI NormalizeIntensityd")
                # Added percentile clip because the max value were being too big!
                if self.mri_clip_percentile:
                    print(f"Clip MRI Percentile")
                    val_transforms_list.append(ClipIntensityPercentilesd(
                                                    keys=[self.key_in],
                                                    lower=0.1,
                                                    upper=99.9
                                                    )
                    )
                val_transforms_list.append(NormalizeIntensityd(
                                                keys=[self.key_in],
                                                subtrahend=None,
                                                divisor=None,
                                                nonzero=False,
                                                channel_wise=False
                                                )
                    ) 
            elif self.data_norm_mri == 'NormalizeIntensityd_Scaled': 
                print(f"Doing MRI NormalizeIntensityd_Scaled")
                # Added percentile clip because the max value were being too big!
                if self.mri_clip_percentile:
                    print(f"Clip MRI Percentile")
                    val_transforms_list.append(ClipIntensityPercentilesd(
                                                    keys=[self.key_in],
                                                    lower=0.1,
                                                    upper=99.9
                                                    )
                    )
                val_transforms_list.append(NormalizeIntensityd(
                                                keys=[self.key_in],
                                                subtrahend=None,
                                                divisor=None,
                                                nonzero=False,
                                                channel_wise=False
                                                )
                ) 
                val_transforms_list.append(ScaleIntensityd(
                                                keys=[self.key_in],
                                                minv=-1.0,
                                                maxv=1.0,
                                                )
                                        )
            else:
                raise ValueError(f"Unsupported data normalization method: {self.data_norm_mri}. Available ScaleIntensityRanged (linear, default) or 'NormalizeIntensityd' (z-score) or NormalizeIntensityd_Scaled (z-score and linear to -1 and 1)")
            
        val_transforms_list.append(CopyItemsd(keys=[self.key_in,self.key_out], times=1, names=[f"{self.key_in}_crop", f"{self.key_out}_crop"], allow_missing_keys=False))
        val_transforms_list.append(RandSpatialCropSamplesd(
                                            keys=[f"{self.key_in}_crop", f"{self.key_out}_crop"],
                                            roi_size=self.spatial_size,
                                            num_samples=self.patch_num,
                                            random_size=False)
                                        )

        val_transforms_list.append(ResizeWithPadOrCropd(
                                            keys=[f"{self.key_in}_crop", f"{self.key_out}_crop"],
                                            spatial_size=self.spatial_size,
                                            mode="minimum")
                                            #value=-1)
                                        )
        
        val_transforms_list.append(ToTensord(keys=[self.key_in, self.key_out, self.key_mask]))
        val_transforms = Compose(val_transforms_list)
        
        return val_transforms

    def get_dataset(self):
        """
        Creates and returns a CacheDataset with the specified transforms.
        
        Returns:
            A CacheDataset instance.
        """
        train_transforms = self.get_train_transforms()
        train_dataset = CacheDataset(data=self.train_data_files, transform=train_transforms, cache_rate=self.cache_rate, num_workers=self.num_workers)

        val_transforms = self.get_val_transforms()
        val_dataset = CacheDataset(data=self.val_data_files, transform=val_transforms, cache_rate=self.cache_rate, num_workers=self.num_workers)
        return train_dataset, val_dataset, train_transforms, val_transforms

    def get_dataloader(self):
        """
        Creates and returns a DataLoader from the CacheDataset.
        
        Returns:
            A DataLoader instance.
        """
        train_dataset, val_dataset, train_transforms, val_transforms = self.get_dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=pad_list_data_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=pad_list_data_collate) 

        return train_dataloader, val_dataloader, train_transforms, val_transforms

    def create_stats_file(self, file_paths, a_min_ct, a_max_ct, normalization_stats_path):
        # Define MONAI transforms
        transforms = [
            LoadImaged(keys=["ct", "mask", "mri"]),
            CropForegroundd(keys=["ct", "mask", "mri"], source_key="mask", allow_smaller=False),  # Crop based on mask
        ]

        print(f"Training cases: {len(file_paths)}")
        # Load dataset
        dataset = Dataset(data=file_paths, transform=transforms)
        dataloader = DataLoader(dataset, batch_size=1)  # Process one patient at a time

        sum_ct    = 0.0
        sum_sq_ct = 0.0
        n_ct      = 0

        sum_mri    = 0.0
        sum_sq_mri = 0.0
        n_mri      = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Creating stats json file for training set")):
            # get raw tensors → move to CPU → convert once to numpy
            ct  = batch['ct'].cpu().numpy()   # shape (B,1,H,W,D)
            mri = batch['mri'].cpu().numpy()
            mask = batch['mask'].cpu().numpy()

            # clip CT in HU
            ct = np.clip(ct, a_min_ct, a_max_ct)
            # Foreground only
            ct = ct[(mask > 0)]

            # flatten to 1D
            ct_flat  = ct.ravel()
            sum_ct    += ct_flat.sum()
            sum_sq_ct += (ct_flat**2).sum()
            n_ct      += ct_flat.size

            # MRI (if you really want a global MRI mean/std – though you’ll use per-scan at inference)
            # Foreground only
            mri = mri[mask > 0]  
            mri_flat  = mri.ravel()
            sum_mri    += mri_flat.sum()
            sum_sq_mri += (mri_flat**2).sum()
            n_mri      += mri_flat.size

        # compute CT mean/std
        mean_ct = sum_ct    / n_ct
        std_ct  = np.sqrt(sum_sq_ct/n_ct - mean_ct**2)

        # (optional) global MRI mean/std
        mean_mri = sum_mri    / n_mri
        std_mri  = np.sqrt(sum_sq_mri/n_mri - mean_mri**2)
        print(f"mean_ct: {mean_ct}")
        print(f"std_ct: {std_ct}")
        print(f"mean_mri: {mean_mri}")
        print(f"std_mri: {std_mri}")
        stats = {
            "ct_mean":  float(mean_ct),
            "ct_std":   float(std_ct),
            "mri_mean": float(mean_mri),
            "mri_std":  float(std_mri),
        }

        with open(normalization_stats_path, "w") as f:
            json.dump(stats, f, indent=4)

        print(f"CT μ={mean_ct:.3f}, σ={std_ct:.3f}")
        print(f"MRI μ={mean_mri:.3f}, σ={std_mri:.3f}")

if __name__ == '__main__':
    DATASET_PATH = "../Dataset"
    ### Create list of cases to load
    data_list_task_1 = []

    for folder in listdir(DATASET_PATH):
        if "Task1" in folder and "Train" in folder:
            sub_fold_train_path = join(DATASET_PATH, folder, "Task1")
            if not isdir(sub_fold_train_path):
                continue
            for intitution in listdir(sub_fold_train_path):
                institution_data_path = join(sub_fold_train_path, intitution)
                if not isdir(institution_data_path):
                    continue            
                for patient in listdir(institution_data_path):
                    if patient=="overviews":
                        continue
                    patient_path = join(institution_data_path, patient)
                    ct_path = join(patient_path, "ct.mha")
                    mask_path = join(patient_path, "mask.mha")
                    mr_path = join(patient_path, "mr.mha")
                    data_list_task_1.append({"ct": ct_path, "mask": mask_path, "mri": mr_path})

    print(f"{len(data_list_task_1)} cases. {data_list_task_1[0]}")


    # Initialize DataLoader with a batch size of 1 and spatial size of 32
    data_loader = MonaiDataLoader(train_data_files=data_list_task_1, val_data_files=data_list_task_1, spatial_size=(32, 32, 32), batch_size=4, shuffle=False)
    train_dataloader, val_dataloader = data_loader.get_dataloader()

    # Iterate over the dataloader and check the shape of the output
    for batch in train_dataloader:
        mri, ct = batch[data_loader.key_in], batch[data_loader.key_out]

        # Assert that the output tensors have the correct shape
        print(f"mri: {mri.shape}")  # [B, C, H, W, D]
        print(f"ct: {ct.shape}")  # [B, C, H, W, D]
        break  

    for batch in val_dataloader:
        mri, ct = batch[data_loader.key_in], batch[data_loader.key_out]

        # Assert that the output tensors have the correct shape
        print(f"mri: {mri.shape}")  # [B, C, H, W, D]
        print(f"ct: {ct.shape}")  # [B, C, H, W, D]
        break  

    print("Data loader good to go!")