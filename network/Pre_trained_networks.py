import os
import re
import torch
#from monai.networks.nets import SwinUNETR
os.environ['nnUNet_raw'] = ''
os.environ['nnUNet_preprocessed'] = ''
os.environ['nnUNet_results'] = ''
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))
from Swin_UNETR_DDPM import SwinUNETR as SwinUNETR_DDPM

def test_forward_pass(model, dimensions=(1, 1, 96, 96, 96), expected_out_channels=1):
    print('\n--- Testing Forward Pass ---')
    dummy_input = torch.randn(*dimensions).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print('Forward pass successful!')
        print(f'Output shape: {output.shape}')
        expected_shape = (dimensions[0], expected_out_channels, dimensions[2], dimensions[3], dimensions[4])
        assert output.shape == expected_shape, f'Expected output shape {expected_shape}, but got {output.shape}'
        print('✅ All good.')

def test_forward_pass_t(model, dimensions=(1, 1, 96, 96, 96), expected_out_channels=1):
    print('\n--- Testing Forward Pass ---')
    dummy_input = torch.randn(*dimensions).to(device)
    B = 1
    T = 1000
    t = torch.randint(0, T, (B,), dtype=torch.long)
    t = t.cuda()
    with torch.no_grad():
        output = model(dummy_input, t)
        print('Forward pass successful!')
        print(f'Output shape: {output.shape}')
        expected_shape = (dimensions[0], expected_out_channels, dimensions[2], dimensions[3], dimensions[4])
        assert output.shape == expected_shape, f'Expected output shape {expected_shape}, but got {output.shape}'
        print('✅ All good.')

def semi_load_weights(model, pretrained_dict, verbose):
    """
    Load weights into a model from a pretrained state_dict, filtering out keys with mismatched shapes,
    and freeze the layers that are loaded.

    Args:
        model (torch.nn.Module): The model to load weights into.
        pretrained_dict (dict): The state_dict containing pretrained weights.

    Returns:
        torch.nn.Module: The model with the loaded weights and frozen layers.
    """
    model_dict = model.state_dict()
    print(f'Total number of features in model: {len(model_dict)}')
    print(f'Total number of features in pretrained model: {len(pretrained_dict)}')
    not_matching_shape_keys_in_pretrained = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape != model_dict[k].shape}
    not_matching_shape_keys_in_model = {k: v for k, v in model_dict.items() if k in pretrained_dict and v.shape != pretrained_dict[k].shape}
    not_matching_keys_in_model = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
    not_matching_keys_in_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in model_dict}
    matching_keys = {k: v for k, v in model_dict.items() if k in pretrained_dict and v.shape == pretrained_dict[k].shape}
    not_matching_keys = {k: v for k, v in model_dict.items() if k not in pretrained_dict or v.shape != pretrained_dict[k].shape}
    print(f'\n✅ Total matching: {len(matching_keys)}')
    print(f'\n❌ Total not matching: {len(not_matching_keys)}')
    if verbose:
        print(f'\n✅ Keys and shapes matching ({len(matching_keys)}):')
        for k in sorted(matching_keys):
            print(f'  {k}')
        print(f'\n❌ Keys in pretained dict that are in model dict but not matching shape (probably input and output layers and timesteps) ({len(not_matching_shape_keys_in_pretrained)}):')
        for k in sorted(not_matching_shape_keys_in_pretrained):
            print(f'  {k}')
        print(f'\n❌ Keys in model dict that are in pretrained dict but not matching shape (probably input and output layers and timesteps) ({len(not_matching_shape_keys_in_model)}):')
        for k in sorted(not_matching_shape_keys_in_model):
            print(f'  {k}')
        print(f'\n❌ Keys in the model dict but not in the pretrained dict ({len(not_matching_keys_in_model)}):')
        for k in sorted(not_matching_keys_in_model):
            print(f'  {k}')
        print(f'\n❌ Keys in the pretrained dict but not in the model dict ({len(not_matching_keys_in_pretrained_dict)}):')
        for k in sorted(not_matching_keys_in_pretrained_dict):
            print(f'  {k}')
    model_dict.update(matching_keys)
    model.load_state_dict(model_dict)
    return (model, matching_keys, not_matching_keys)

def freeze_layers(model, matching_keys, not_matching_keys):
    """
    Freeze the layers of the model that were loaded from the pretrained weights.

    Args:
        model (torch.nn.Module): The model to freeze layers in.
        matching_keys (dict): The state_dict containing pretrained weights.

    Returns:
        None
    """
    for name, param in model.named_parameters():
        if name in matching_keys:
            param.requires_grad = False
    print('Frozen layers loaded from pretrained weights.')
    print('NOT Frozen new layers:')
    for name, param in model.named_parameters():
        if name in not_matching_keys:
            print(name)
            param.requires_grad = True
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print('Trainable parameters:', trainable_params)
    return model

def fix_checkpoint_keys_swinUNETR(state_dict):
    """
    This function replaces the 'module.' with 'swinViT.' 
    and the linear layers '.mlp.fc' with '.mlp.linear'
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('module.'):
            new_key = key.replace('module.', 'swinViT.')
            new_key = new_key.replace('.mlp.fc', '.mlp.linear')
        fixed_state_dict[new_key] = value
    return fixed_state_dict

def load_pretrained_swinvit(ckpt_path, load_weights, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, use_checkpoint=True, verbose=False):
    """
    Load a pretrained SwinViT model from a checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        img_size (tuple): Image size for the model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        feature_size (int): Feature size for the model.
        use_checkpoint (bool): Whether to use checkpointing.

    Returns:
        model: The loaded SwinUNETR model.
    """
    model_dict = torch.load(ckpt_path, weights_only=False)
    state_dict = model_dict.get('state_dict', model_dict)
    state_dict = fix_checkpoint_keys_swinUNETR(state_dict)
    model = SwinUNETR_DDPM(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=feature_size, use_checkpoint=use_checkpoint)
    if load_weights:
        model, matching_keys, not_matching_keys = semi_load_weights(model=model, pretrained_dict=state_dict, verbose=verbose)
        print('✅ Model weights loaded (non-strict mode).')
    return (model, matching_keys, not_matching_keys)

def load_pretrained_SwinUNETR(ckpt_path, load_weights, img_size, in_channels, out_channels, feature_size, use_checkpoint, verbose=False):
    """
    Load a pretrained SwinUNETR model from a checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        img_size (tuple): Image size for the model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        feature_size (int): Feature size for the model.
        use_checkpoint (bool): Whether to use checkpointing.

    Returns:
        model: The loaded SwinUNETR model.
    """
    model = SwinUNETR_DDPM(img_size=img_size, in_channels=in_channels, out_channels=out_channels, feature_size=feature_size, use_checkpoint=use_checkpoint)
    
    if load_weights:
        model_dict = torch.load(ckpt_path, weights_only=False)
        state_dict = model_dict.get('state_dict', model_dict)
        model, matching_keys, not_matching_keys = semi_load_weights(model=model, pretrained_dict=state_dict, verbose=verbose)
        print('✅ Model weights loaded (non-strict mode).')
    else:
        matching_keys = None
        not_matching_keys = None
    return (model, matching_keys, not_matching_keys)

def fix_transpconv_keys_nnunet(pretrained_state_dict):
    """
    Adapts keys from a pretrained state_dict to match a new model
    structure and filters out unwanted keys.

    Specifically:
    1. Transforms 'decoder.stages.<idx>.<rest>' keys to
       'decoder.stages.<idx>.0.<rest>'
    2. Transforms 'decoder.transpconvs.<idx>.<rest>' keys to
       'decoder.transpconvs.<idx>.0.<rest>'
    3. Filters out (removes) any keys containing "all_modules".
    4. Keeps other keys unchanged.

    Args:
        pretrained_state_dict (dict): The state dictionary loaded from the
                                      pre-trained model file.

    Returns:
        dict: A new state dictionary adapted and filtered for the new model.
    """
    new_state_dict = {}
    adaptation_pattern = re.compile('^(decoder\\.(?:stages|transpconvs)\\.\\d+)(\\..*)')
    print('Adapting and filtering state dict (handling stages and transpconvs)...')
    adapted_count = 0
    filtered_count = 0
    kept_unchanged_count = 0
    for k, v in pretrained_state_dict.items():
        if 'all_modules' in k:
            filtered_count += 1
            continue
        match = adaptation_pattern.match(k)
        if match:
            new_k = match.group(1) + '.0' + match.group(2)
            new_state_dict[new_k] = v
            adapted_count += 1
        else:
            new_state_dict[k] = v
            if not k.startswith('decoder.'):
                kept_unchanged_count += 1
    print('Finished processing state dict:')
    print(f"- Adapted {adapted_count} 'decoder.stages' or 'decoder.transpconvs' keys.")
    print(f"- Filtered out {filtered_count} keys containing 'all_modules'.")
    return new_state_dict

def load_pretrained_TotalSegmentator(ckpt_path, load_weights, verbose):
    """
    Load a pretrained TotalSegmentator model from a checkpoint.

    Args:
        ckpt_path (str): Path to the checkpoint directory.
        verbose (bool): Whether to print detailed information about the loading process.

    Returns:
        model: The loaded TotalSegmentator model.
    """
    os.environ['nnUNet_raw'] = ''
    os.environ['nnUNet_preprocessed'] = ''
    os.environ['nnUNet_results'] = ''
    checkpoint_name = 'checkpoint_final.pth'
    print(f"ckpt_path: {ckpt_path}")
    try:
        ckpt_path = ckpt_path.split('/fold_0')[0]
    except:
        print(f"ckpt_path: {ckpt_path}")
        raise Exception('The checkpoint_final.pth needs to be inside of the folder fold_0')
    dataset_json = load_json(join(ckpt_path, 'dataset.json'))
    plans = load_json(join(ckpt_path, 'plans.json'))
    plans_manager = PlansManager(plans)
    use_folds = [0]
    if isinstance(use_folds, str):
        use_folds = [use_folds]
    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(ckpt_path, f'fold_{f}', checkpoint_name), map_location=torch.device('cpu'), weights_only=False)
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)
        parameters.append(checkpoint['network_weights'])
    new_state_dict = fix_transpconv_keys_nnunet(parameters[0])
    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'), trainer_name, 'nnunetv2.training.nnUNetTrainer')
    model = trainer_class.build_network_architecture(configuration_manager.network_arch_class_name, configuration_manager.network_arch_init_kwargs, configuration_manager.network_arch_init_kwargs_req_import, num_input_channels, plans_manager.get_label_manager(dataset_json).num_segmentation_heads, enable_deep_supervision=False)
    if load_weights:
        model, matching_keys, not_matching_keys = semi_load_weights(model=model, pretrained_dict=new_state_dict, verbose=verbose)
        print('✅ Model weights loaded (non-strict mode).')
    else:
        matching_keys = None
        not_matching_keys = None
    return (model, matching_keys, not_matching_keys)

if __name__ == '__main__':
    device = 'cuda'
    """
    # Test model_swinvit
    print('Test model_swinvit')
    swinvit_ckpt_path = join(
        '/projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/SwinUNETR/model_swinvit.pt'
    )
    model, matching_keys, not_matching_keys = load_pretrained_swinvit(
        ckpt_path=swinvit_ckpt_path,
        load_weights=True,
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        verbose=False
    )
    model = freeze_layers(model=model, matching_keys=matching_keys, not_matching_keys=not_matching_keys)
    test_forward_pass_t(model.cuda(), dimensions=(1, 2, 96, 96, 96), expected_out_channels=2)

    # Test model_swinUNETR
    print('Test model_swinUNETR')
    swinunetr_ckpt_path = join(
        '/projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/SwinUNETR/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt'
    )
    model, matching_keys, not_matching_keys = load_pretrained_SwinUNETR(
        ckpt_path=swinunetr_ckpt_path,
        load_weights=True,
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        verbose=False
    )
    model = freeze_layers(model=model, matching_keys=matching_keys, not_matching_keys=not_matching_keys)
    test_forward_pass_t(model.cuda(), dimensions=(1, 2, 96, 96, 96), expected_out_channels=2)

    # Test TotalSegmentator light
    print('Test TotalSegmentator light')
    totalsegmentator_light_ckpt_path = '/projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/TotalSegmentator/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
    model, matching_keys, not_matching_keys = load_pretrained_TotalSegmentator(
        ckpt_path=totalsegmentator_light_ckpt_path,
        load_weights=True,
        verbose=False
    )
    model = freeze_layers(model=model, matching_keys=matching_keys, not_matching_keys=not_matching_keys)
    test_forward_pass_t(model.cuda(), dimensions=(1, 2, 128, 128, 128), expected_out_channels=2)

    # Test TotalSegmentator full res organs
    print('Test TotalSegmentator full res organs')
    totalsegmentator_organs_ckpt_path = '/projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/TotalSegmentator/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
    model, matching_keys, not_matching_keys = load_pretrained_TotalSegmentator(
        ckpt_path=totalsegmentator_organs_ckpt_path,
        load_weights=True,
        verbose=False
    )
    model = freeze_layers(model=model, matching_keys=matching_keys, not_matching_keys=not_matching_keys)
    test_forward_pass_t(model.cuda(), dimensions=(1, 2, 128, 128, 128), expected_out_channels=2)
    """
    # Test TotalSegmentator mri model 
    print('Test TotalSegmentator MRI model')
    totalsegmentator_muscles_ckpt_path = '/projects/nian/synthrad2025/src/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model/network/pre_trained/TotalSegmentator/mri/Dataset852_TotalSegMRI_total_3mm_1088subj/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
    model, matching_keys, not_matching_keys = load_pretrained_TotalSegmentator(
        ckpt_path=totalsegmentator_muscles_ckpt_path,
        load_weights=True,
        verbose=True
    )
    model = freeze_layers(model=model, matching_keys=matching_keys, not_matching_keys=not_matching_keys)
    test_forward_pass_t(model.cuda(), dimensions=(1, 2, 128, 128, 32), expected_out_channels=2)