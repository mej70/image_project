import os
import numpy as np
import matplotlib.pyplot as plt

from functions import find_dwi_datasets, load_dwi_dataset, compute_dti, tensor_to_6d, compute_color_fa_from_tensor6

from dipy.denoise.patch2self import patch2self

def plot_original_denoising(original, denoised, bvals, gtab, z_slice=12):
    dwi_indices = np.where(bvals > 500)[0]
    vol_idx = dwi_indices[0] if len(dwi_indices) > 0 else 0

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def prep_img(img):
        return np.rot90(img)

    # --- ZEILE 1: Rohdaten ---
    vmax = np.percentile(original[:, :, z_slice, vol_idx], 99)
    axes[0, 0].imshow(prep_img(original[:, :, z_slice, vol_idx]), cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title('Ground Truth (Raw MRI)')
    
    axes[0, 1].imshow(prep_img(denoised[:, :, z_slice, vol_idx]), cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title('Denoised (Patch2Self)')

    diff_error = np.abs(original[:, :, z_slice, vol_idx] - denoised[:, :, z_slice, vol_idx])
    res_vmax = np.percentile(diff_error, 99)
    axes[0, 2].imshow(prep_img(diff_error), cmap='hot', vmin=0, vmax=res_vmax)
    axes[0, 2].set_title('Denoised Heatmap (|Ground Truth - Denoised|)')

    # --- ZEILE 2: Color-FA Maps ---
    orig_slice = original[:, :, z_slice:z_slice+1, :]
    den_slice = denoised[:, :, z_slice:z_slice+1, :]

    fa_orig = compute_color_fa_from_tensor6(tensor_to_6d(compute_dti(orig_slice, gtab)))[:, :, 0, :]
    fa_den = compute_color_fa_from_tensor6(tensor_to_6d(compute_dti(den_slice, gtab)))[:, :, 0, :]

    fa_orig = np.clip(fa_orig, 0, 1)
    fa_den = np.clip(fa_den, 0, 1)

    axes[1, 0].imshow(prep_img(fa_orig))
    axes[1, 0].set_title('Color-FA Map (Ground Truth)')
    
    axes[1, 1].imshow(prep_img(fa_den))
    axes[1, 1].set_title('Color-FA Map (Denoised)')
    
    axes[1, 2].axis('off')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'denoising_original.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'example_dti'))
    entries = find_dwi_datasets(data_dir)
    
    if len(entries) == 0:
        print(f"No data in {data_dir}")
        return

    sample = load_dwi_dataset(entries[0])
    original_data = sample['data']
    bvals = sample['bvals']
    gtab = sample['gtab']

    denoised_data = patch2self(original_data, bvals, shift_intensity=True, clip_negative_vals=False, b0_threshold=50, version=3)

    mid_z = original_data.shape[2] // 2
    plot_original_denoising(original_data, denoised_data, bvals, gtab, z_slice=mid_z)

if __name__ == "__main__":
    main()
