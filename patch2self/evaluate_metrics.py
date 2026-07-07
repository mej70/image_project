import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Append U-Net directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'U-Net')))
from UnetCore import UNet

from functions import (
    find_dwi_datasets, load_dwi_dataset, norm,
    compute_dti, tensor_to_6d, compute_md_from_tensor6, compute_fa_from_tensor6, compute_color_fa_from_tensor6, brain_mask,
    ssim_pytorch
)
from denoise_unet import infer_masked_patch2self


def compute_volume_ssim(original, denoised, mask=None):
    """
    Computes the average 2D SSIM across all volumes and slices in the datasets,
    matching the structural similarity calculation in the training loss.
    """
    if mask is not None:
        if len(original.shape) == 4:
            original = original * mask[..., np.newaxis]
            denoised = denoised * mask[..., np.newaxis]
        else:
            original = original * mask
            denoised = denoised * mask
            
    orig_tensor = torch.tensor(original, dtype=torch.float32)
    den_tensor = torch.tensor(denoised, dtype=torch.float32)
    
    if len(orig_tensor.shape) == 3: # X, Y, Z
        X, Y, Z = orig_tensor.shape
        orig_2d = orig_tensor.permute(2, 0, 1).reshape(Z, 1, X, Y)
        den_2d = den_tensor.permute(2, 0, 1).reshape(Z, 1, X, Y)
    else: # X, Y, Z, N
        X, Y, Z, N = orig_tensor.shape
        orig_2d = orig_tensor.permute(3, 2, 0, 1).reshape(N * Z, 1, X, Y)
        den_2d = den_tensor.permute(3, 2, 0, 1).reshape(N * Z, 1, X, Y)
    
    d_range = float(orig_tensor.max() - orig_tensor.min())
    if d_range == 0:
        d_range = 1.0
        
    score = ssim_pytorch(orig_2d, den_2d, data_range=d_range)
    return score.item()


def get_md_fa(data, gtab, mask):
    tensor = compute_dti(data, gtab)
    tensor6 = tensor_to_6d(tensor)
    md = compute_md_from_tensor6(tensor6)
    fa = compute_fa_from_tensor6(tensor6)
    
    if mask is not None:
        md_masked = md[mask > 0]
        fa_masked = fa[mask > 0]
        return md, fa, md_masked, fa_masked
    else:
        return md, fa, md.ravel(), fa.ravel()


def evaluate_single_config(entry, sample, model_path=None, baseline=None, precomputed_path=None, shuffle_volumes=False, denoise_b0=False, scale_intensities=False, device='cuda'):
    raw_data = sample['data']
    bvals = sample['bvals']
    gtab = sample['gtab']
    N_volumes = raw_data.shape[-1]
    
    # Get brain mask
    mask, _ = brain_mask(raw_data[..., 0])
    
    dataset_name = os.path.basename(entry["dwi"]).replace(".nii.gz", "")
    
    # Detect loss subdirectory if model is inside patch2self/trained/<loss_type>
    loss_subdir = ""
    if model_path is not None:
        parent_dir = os.path.dirname(os.path.abspath(model_path))
        grandparent_dir = os.path.dirname(parent_dir)
        if os.path.basename(grandparent_dir) == 'trained':
            loss_subdir = os.path.basename(parent_dir)
            
    denoised_data = None
    model_name_label = ""
    
    if baseline == "patch2self":
        from dipy.denoise.patch2self import patch2self
        print(f"\n--- Running DIPY Patch2Self Baseline (Masked) on {dataset_name} ---")
        masked_raw = raw_data * mask[..., np.newaxis]
        denoised_data = patch2self(masked_raw, bvals, shift_intensity=True, clip_negative_vals=False, b0_threshold=50, version=3)
        denoised_data = denoised_data * mask[..., np.newaxis]
        model_name_label = "Patch2Self"
        
    elif baseline == "bm4d":
        import bm4d
        from dipy.denoise.noise_estimate import estimate_sigma
        print(f"\n--- Running BM4D Baseline (Masked) on {dataset_name} ---")
        denoised_data = np.zeros_like(raw_data)
        masked_raw = raw_data * mask[..., np.newaxis]
        for v in range(N_volumes):
            vol_raw = raw_data[..., v].astype(np.float32)
            vol_masked = masked_raw[..., v].astype(np.float32)
            sigma = np.mean(estimate_sigma(vol_raw))
            denoised_data[..., v] = bm4d.bm4d(vol_masked, sigma_psd=sigma)
        denoised_data = denoised_data * mask[..., np.newaxis]
        model_name_label = "BM4D"
        
    elif precomputed_path:
        import nibabel as nib
        print(f"\n--- Loading Precomputed Denoised Data: {precomputed_path} ---")
        denoised_data = nib.load(precomputed_path).get_fdata()
        denoised_data = denoised_data * mask[..., np.newaxis]
        model_name_label = os.path.basename(precomputed_path).replace(".nii.gz", "")
        
    elif model_path:
        model_name_label = os.path.basename(model_path).replace(".pth", "")
        if shuffle_volumes:
            model_name_label += "_Shuffled"
        print(f"\n--- Evaluating U-Net Checkpoint: {model_name_label} on {dataset_name} ---")
        
        # Load state dict to auto-detect channels and features
        state_dict = torch.load(model_path, map_location=device)
        weight_keys = [k for k in state_dict.keys() if 'weight' in k]
        first_weight_shape = state_dict[weight_keys[0]].shape
        in_channels = first_weight_shape[1]
        features = first_weight_shape[0]
        
        unet_model = UNet(in_channels=in_channels, out_channels=in_channels, features=features).to(device)
        unet_model.load_state_dict(state_dict)
        unet_model.eval()
        
        # Exclude b0 if model is trained on 128 channels
        if in_channels == 128:
            dwi_indices = np.where(bvals >= 50)[0]
            
            # Apply brain mask to raw data
            masked_raw_data = raw_data * mask[..., np.newaxis]
            dwi_raw_data = masked_raw_data[..., dwi_indices]
            
            normalized_dwi = norm(dwi_raw_data)
            
            # Convert to PyTorch tensor (1, 128, X, Y, Z)
            data_tensor = torch.tensor(normalized_dwi, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            
            if shuffle_volumes:
                # Shuffle DWI channels
                np.random.seed(42)
                shuffled_indices = np.random.permutation(128)
                data_tensor = data_tensor[:, shuffled_indices, :, :, :]
                
            # Run inference
            denoised_normalized_dwi = infer_masked_patch2self(
                data_tensor, 
                (raw_data.shape[0], raw_data.shape[1], raw_data.shape[2], 128), 
                unet_model, 
                device=device
            )
            
            if shuffle_volumes:
                # Unshuffle
                unshuffled_denoised = np.zeros_like(denoised_normalized_dwi)
                for new_idx, old_idx in enumerate(shuffled_indices):
                    unshuffled_denoised[..., old_idx] = denoised_normalized_dwi[..., new_idx]
                denoised_normalized_dwi = unshuffled_denoised
                
            # Denormalize
            dwi_raw = raw_data[..., dwi_indices]
            masked_dwi_raw = dwi_raw * mask[..., np.newaxis]
            vmin, vmax = np.percentile(masked_dwi_raw[masked_dwi_raw > 0], (1, 99))
            denoised_dwi = denoised_normalized_dwi * (vmax - vmin + 1e-8) + vmin
            
            # Reassemble (keep b0 raw)
            denoised_data = raw_data.copy()
            denoised_data[..., dwi_indices] = denoised_dwi
            
        else:
            # 130 channel model
            masked_raw_data = raw_data * mask[..., np.newaxis]
            normalized_data = norm(masked_raw_data)
            data_tensor = torch.tensor(normalized_data, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            
            if shuffle_volumes:
                np.random.seed(42)
                shuffled_indices = np.random.permutation(N_volumes)
                data_tensor = data_tensor[:, shuffled_indices, :, :, :]
                
            denoised_normalized_data = infer_masked_patch2self(
                data_tensor, 
                raw_data.shape, 
                unet_model, 
                device=device
            )
            
            if shuffle_volumes:
                unshuffled_denoised = np.zeros_like(denoised_normalized_data)
                for new_idx, old_idx in enumerate(shuffled_indices):
                    unshuffled_denoised[..., old_idx] = denoised_normalized_data[..., new_idx]
                denoised_normalized_data = unshuffled_denoised
                
            vmin, vmax = np.percentile(masked_raw_data[masked_raw_data > 0], (1, 99))
            denoised_data = denoised_normalized_data * (vmax - vmin + 1e-8) + vmin
    if denoise_b0:
        print("\n--- Denoising b0 volumes using Non-Local Means (NL-Means) ---")
        from dipy.denoise.nlmeans import nlmeans
        from dipy.denoise.noise_estimate import estimate_sigma
        b0_indices = np.where(bvals < 50)[0]
        for idx in b0_indices:
            vol = denoised_data[..., idx].astype(np.float64)
            raw_vol = raw_data[..., idx].astype(np.float64)
            sigma = estimate_sigma(raw_vol)
            denoised_b0_vol = nlmeans(vol, sigma=sigma, mask=mask, rician=True)
            denoised_data[..., idx] = denoised_b0_vol
        # Force background to 0
        denoised_data = denoised_data * mask[..., np.newaxis]

    if scale_intensities:
        dwi_indices = np.where(bvals >= 50)[0]
        raw_dwi_masked = raw_data[..., dwi_indices] * mask[..., np.newaxis]
        den_dwi_masked = denoised_data[..., dwi_indices] * mask[..., np.newaxis]
        
        mean_raw = np.mean(raw_dwi_masked[raw_dwi_masked > 0]) if np.any(raw_dwi_masked > 0) else 1.0
        mean_den = np.mean(den_dwi_masked[den_dwi_masked > 0]) if np.any(den_dwi_masked > 0) else 1.0
        
        alpha = mean_raw / (mean_den + 1e-8)
        print(f"\n--- Scaling Intensities to match Raw: alpha = {alpha:.4f} ---")
        denoised_data[..., dwi_indices] = denoised_data[..., dwi_indices] * alpha
        denoised_data = denoised_data * mask[..., np.newaxis]

    # Calculate DTI parameters
    md_orig, fa_orig, md_orig_m, fa_orig_m = get_md_fa(raw_data, gtab, mask)
    md_denoised, fa_denoised, md_denoised_m, fa_denoised_m = get_md_fa(denoised_data, gtab, mask)
    
    valid_orig = np.isfinite(md_orig_m) & np.isfinite(fa_orig_m)
    valid_denoised = np.isfinite(md_denoised_m) & np.isfinite(fa_denoised_m)
    valid_idx = valid_orig & valid_denoised
    
    md_orig_v = md_orig_m[valid_idx]
    fa_orig_v = fa_orig_m[valid_idx]
    md_denoised_v = md_denoised_m[valid_idx]
    fa_denoised_v = fa_denoised_m[valid_idx]
    
    md_mae = np.mean(np.abs(md_orig_v - md_denoised_v))
    fa_mae = np.mean(np.abs(fa_orig_v - fa_denoised_v))
    md_wasserstein = wasserstein_distance(md_orig_v, md_denoised_v)
    fa_wasserstein = wasserstein_distance(fa_orig_v, fa_denoised_v)
    ssim_val = compute_volume_ssim(raw_data, denoised_data, mask)
    
    print("\n--- Metrics ---")
    print(f"MD MAE: {md_mae:.6f}")
    print(f"FA MAE: {fa_mae:.6f}")
    print(f"MD Wasserstein Distance: {md_wasserstein:.6f}")
    print(f"FA Wasserstein Distance: {fa_wasserstein:.6f}")
    print(f"Volume SSIM (vs Original): {ssim_val:.6f}")
    
    # Save metrics JSON
    metrics_dict = {
        "model": model_name_label,
        "dataset": dataset_name,
        "md_mae": float(md_mae),
        "fa_mae": float(fa_mae),
        "md_wasserstein": float(md_wasserstein),
        "fa_wasserstein": float(fa_wasserstein),
        "volume_ssim": float(ssim_val)
    }
    
    if loss_subdir:
        metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics', loss_subdir, dataset_name)
    else:
        metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics', dataset_name)
    os.makedirs(metrics_dir, exist_ok=True)
    safe_name = model_name_label.replace('.pth', '').replace(' ', '_')
    if denoise_b0:
        safe_name += "_denoise_b0"
    if scale_intensities:
        safe_name += "_scaled"
    json_path = os.path.join(metrics_dir, f"{safe_name}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics to {json_path}")
    
    # Generate Histograms
    print("Generating Histograms...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    md_p99 = np.percentile(md_orig_m, 99.5)
    bins_md = np.linspace(0, md_p99, 100)
    axs[0].hist(md_orig_m, bins=bins_md, color='blue', alpha=0.5, label='Original', density=True)
    axs[0].hist(md_denoised_m, bins=bins_md, color='red', alpha=0.5, label=model_name_label, density=True)
    axs[0].set_title(f"MD Distribution\nWD: {md_wasserstein:.4f}")
    axs[0].set_xlabel("Mean Diffusivity")
    axs[0].set_ylabel("Density")
    axs[0].legend()
    
    bins_fa = np.linspace(0, 1.0, 100)
    axs[1].hist(fa_orig_m, bins=bins_fa, color='blue', alpha=0.5, label='Original', density=True)
    axs[1].hist(fa_denoised_m, bins=bins_fa, color='red', alpha=0.5, label=model_name_label, density=True)
    axs[1].set_title(f"FA Distribution\nWD: {fa_wasserstein:.4f}")
    axs[1].set_xlabel("Fractional Anisotropy")
    axs[1].set_ylabel("Density")
    axs[1].legend()
    
    plt.tight_layout()
    if loss_subdir:
        plots_dir = os.path.join(os.path.dirname(__file__), 'unet_plots', loss_subdir, dataset_name)
    else:
        plots_dir = os.path.join(os.path.dirname(__file__), 'unet_plots', dataset_name)
    os.makedirs(plots_dir, exist_ok=True)
    hist_path = os.path.join(plots_dir, f"{safe_name}_metrics_histograms.png")
    plt.savefig(hist_path, dpi=300)
    plt.close(fig)
    print(f"Saved Histograms to {hist_path}")
    
    # Generate 3x3 Visual Plot
    print("Generating 3x3 Visual Plot (Raw, FA, MD)...")
    z_slice = raw_data.shape[2] // 2
    dwi_indices = np.where(bvals > 500)[0]
    vol_idx = dwi_indices[0] if len(dwi_indices) > 0 else 0
    
    mask_slice = mask[:, :, z_slice]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    def prep_img(img):
        return np.rot90(img)
        
    # Masked Raw MRI
    raw_img = raw_data[:, :, z_slice, vol_idx] * mask_slice
    den_img = denoised_data[:, :, z_slice, vol_idx] * mask_slice
    diff_error = np.abs(raw_img - den_img)
    
    vmax_raw = np.percentile(raw_img[mask_slice > 0], 99) if len(raw_img[mask_slice > 0]) > 0 else raw_img.max()
    axes[0, 0].imshow(prep_img(raw_img), cmap='gray', vmin=0, vmax=vmax_raw)
    axes[0, 0].set_title('Ground Truth (Raw MRI, Masked)')
    
    axes[0, 1].imshow(prep_img(den_img), cmap='gray', vmin=0, vmax=vmax_raw)
    axes[0, 1].set_title(f'Denoised ({model_name_label})')
    
    # Use 30% of vmax_raw for difference heatmap to make different models comparable
    res_vmax = vmax_raw * 0.3 if vmax_raw > 0 else 1.0
    axes[0, 2].imshow(prep_img(diff_error), cmap='hot', vmin=0, vmax=res_vmax)
    axes[0, 2].set_title('Denoised Heatmap (|GT - Denoised|)')
    
    orig_slice = raw_data[:, :, z_slice:z_slice+1, :]
    den_slice = denoised_data[:, :, z_slice:z_slice+1, :]
    tensor_orig = tensor_to_6d(compute_dti(orig_slice, gtab))
    tensor_den = tensor_to_6d(compute_dti(den_slice, gtab))
    
    # Color-FA
    fa_c_orig = compute_color_fa_from_tensor6(tensor_orig)[:, :, 0, :] * mask_slice[..., np.newaxis]
    fa_c_den = compute_color_fa_from_tensor6(tensor_den)[:, :, 0, :] * mask_slice[..., np.newaxis]
    axes[1, 0].imshow(prep_img(np.clip(fa_c_orig, 0, 1)))
    axes[1, 0].set_title('Color-FA Map (Ground Truth, Masked)')
    
    axes[1, 1].imshow(prep_img(np.clip(fa_c_den, 0, 1)))
    axes[1, 1].set_title('Color-FA Map (Denoised)')
    
    # Prepare crops for zoomed view (center of the brain, e.g. corpus callosum)
    rot_gt_fa = prep_img(np.clip(fa_c_orig, 0, 1))
    rot_den_fa = prep_img(np.clip(fa_c_den, 0, 1))
    
    H_fa, W_fa = rot_gt_fa.shape[0], rot_gt_fa.shape[1]
    c_h, c_w = H_fa // 2, W_fa // 2
    crop_h, crop_w = int(H_fa * 0.35), int(W_fa * 0.35)
    h_start, h_end = c_h - crop_h // 2, c_h + crop_h // 2
    w_start, w_end = c_w - crop_w // 2, c_w + crop_w // 2
    
    crop_gt_fa = rot_gt_fa[h_start:h_end, w_start:w_end, :]
    crop_den_fa = rot_den_fa[h_start:h_end, w_start:w_end, :]
    
    # Concatenate horizontally with a small black divider
    divider_fa = np.zeros((crop_gt_fa.shape[0], 3, 3), dtype=crop_gt_fa.dtype)
    combined_crop_fa = np.hstack([crop_gt_fa, divider_fa, crop_den_fa])
    
    axes[1, 2].imshow(combined_crop_fa)
    axes[1, 2].set_title('Color-FA Zoom (GT | Denoised)')
    
    # MD
    md_s_orig = compute_md_from_tensor6(tensor_orig)[:, :, 0] * mask_slice
    md_s_den = compute_md_from_tensor6(tensor_den)[:, :, 0] * mask_slice
    md_vmax = np.percentile(md_s_orig[mask_slice > 0], 99) if len(md_s_orig[mask_slice > 0]) > 0 else 0.003
    axes[2, 0].imshow(prep_img(md_s_orig), cmap='gray', vmin=0, vmax=md_vmax)
    axes[2, 0].set_title('Mean Diffusivity (Ground Truth, Masked)')
    
    axes[2, 1].imshow(prep_img(md_s_den), cmap='gray', vmin=0, vmax=md_vmax)
    axes[2, 1].set_title('Mean Diffusivity (Denoised)')
    
    # Prepare MD crops for zoomed view
    rot_gt_md = prep_img(md_s_orig)
    rot_den_md = prep_img(md_s_den)
    
    H_md, W_md = rot_gt_md.shape[0], rot_gt_md.shape[1]
    c_h_md, c_w_md = H_md // 2, W_md // 2
    crop_h_md, crop_w_md = int(H_md * 0.35), int(W_md * 0.35)
    h_start_md, h_end_md = c_h_md - crop_h_md // 2, c_h_md + crop_h_md // 2
    w_start_md, w_end_md = c_w_md - crop_w_md // 2, c_w_md + crop_w_md // 2
    
    crop_gt_md = rot_gt_md[h_start_md:h_end_md, w_start_md:w_end_md]
    crop_den_md = rot_den_md[h_start_md:h_end_md, w_start_md:w_end_md]
    
    divider_md = np.zeros((crop_gt_md.shape[0], 3), dtype=crop_gt_md.dtype)
    combined_crop_md = np.hstack([crop_gt_md, divider_md, crop_den_md])
    
    axes[2, 2].imshow(combined_crop_md, cmap='gray', vmin=0, vmax=md_vmax)
    axes[2, 2].set_title('MD Zoom (GT | Denoised)')
    
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
            
    fig.suptitle(f"Visual Comparison (z-slice: {z_slice})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    vis_path = os.path.join(plots_dir, f"{safe_name}_visual_3x3.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 3x3 Visual Plot to {vis_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Denoising Models with MD/FA Histograms and Wasserstein distance.")
    parser.add_argument("--model", type=str, help="Path to U-Net model (e.g., trained/trained_unet64.pth)")
    parser.add_argument("--model-folder", type=str, help="Path to a directory containing multiple U-Net model checkpoints (.pth)")
    parser.add_argument("--baseline", type=str, choices=["patch2self", "bm4d"], help="Run baseline instead of U-Net")
    parser.add_argument("--precomputed", type=str, help="Path to a pre-denoised .nii.gz file to evaluate directly")
    parser.add_argument("--dataset-index", type=int, default=None, help="Index of a specific dataset to load from example_dti (default: None, evaluates all valid).")
    parser.add_argument("--shuffle-volumes", action="store_true", help="Shuffle input volumes during inference to test channel mapping dependency.")
    parser.add_argument("--denoise-b0", action="store_true", help="Denoise b0 volumes using Non-Local Means (NL-Means) during evaluation.")
    parser.add_argument("--scale-intensities", action="store_true", help="Scale intensities of denoised DWI to match mean raw DWI (prevents MD distribution shift).")
    
    args = parser.parse_args()
    
    if not args.model and not args.model_folder and not args.baseline and not args.precomputed:
        print("Please provide either --model, --model-folder, --baseline, or --precomputed.")
        return
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Find datasets
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example_dti'))
    entries = find_dwi_datasets(data_dir)
    if len(entries) == 0:
        print(f"No data found in {data_dir}")
        return
        
    # We filter datasets to only keep those that match 130 volumes (homogeneous structure)
    valid_entries = []
    import nibabel as nib
    for entry in entries:
        img = nib.load(entry["dwi"])
        c = img.shape[-1]
        if c == 130:
            valid_entries.append(entry)
        else:
            print(f"Skipping {entry['dwi']}: has {c} channels, expected 130.")
            
    if len(valid_entries) == 0:
        print("No valid datasets with 130 channels found.")
        return
        
    # Select which datasets to evaluate
    if args.dataset_index is not None:
        if args.dataset_index >= len(valid_entries):
            print(f"Dataset index {args.dataset_index} out of bounds. Found {len(valid_entries)} valid datasets.")
            return
        datasets_to_eval = [valid_entries[args.dataset_index]]
    else:
        datasets_to_eval = valid_entries
        
    print(f"Evaluating on {len(datasets_to_eval)} datasets.")
    
    # Build list of model configurations
    configs = []
    if args.baseline:
        configs.append({'baseline': args.baseline})
    elif args.precomputed:
        configs.append({'precomputed_path': args.precomputed})
    elif args.model_folder:
        if not os.path.exists(args.model_folder):
            print(f"Model folder not found: {args.model_folder}")
            return
        files = sorted(os.listdir(args.model_folder))
        checkpoint_paths = [os.path.join(args.model_folder, f) for f in files if f.endswith('.pth')]
        if len(checkpoint_paths) == 0:
            print(f"No .pth checkpoint files found in {args.model_folder}")
            return
        print(f"Found {len(checkpoint_paths)} checkpoints in {args.model_folder}.")
        for path in checkpoint_paths:
            configs.append({'model_path': path})
    elif args.model:
        configs.append({'model_path': args.model})
        
    # Load samples
    loaded_samples = []
    for entry in datasets_to_eval:
        sample = load_dwi_dataset(entry)
        loaded_samples.append((entry, sample))
        
    # Evaluate each configuration on each dataset
    for config in configs:
        for entry, sample in loaded_samples:
            try:
                evaluate_single_config(
                    entry, 
                    sample, 
                    model_path=config.get('model_path'), 
                    baseline=config.get('baseline'), 
                    precomputed_path=config.get('precomputed_path'), 
                    shuffle_volumes=args.shuffle_volumes, 
                    denoise_b0=args.denoise_b0,
                    scale_intensities=args.scale_intensities,
                    device=device
                )
            except Exception as e:
                import traceback
                print(f"Failed to evaluate configuration {config} on {entry['dwi']}:")
                traceback.print_exc()


if __name__ == "__main__":
    main()
