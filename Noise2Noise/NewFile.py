import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# --- Settings ---
NOISY_FILE = "../data/sub-02_ses-1_dwi.nii.gz"
DENOISED_FILE = "result.nii.gz"
#DENOISED_FILE = "denoised_result.nii.gz"
BVAL_FILE = "../data/sub-02_ses-1_dwi.bval"  # Required for FA/MD
BVEC_FILE = "../data/sub-02_ses-1_dwi.bvec"  # Required for FA/MD
SLICE_IDX = None  # Set to None for the middle slice


def create_brain_mask(img_3d):
    """Erstellt eine einfache, robuste Gehirnmaske basierend auf Intensitätsschwellenwerten."""
    thresh = np.percentile(img_3d, 70) * 0.3
    mask = img_3d > thresh
    return mask


def calculate_tsnr(data_4d):
    """Berechnet die zeitliche/direktionale Signal-to-Noise Ratio (tSNR)."""
    mean = np.mean(data_4d, axis=-1)
    std = np.std(data_4d, axis=-1)
    tsnr = np.zeros_like(mean)
    valid = std > 0
    tsnr[valid] = mean[valid] / std[valid]
    return tsnr


def calculate_gcor(data_4d, mask):
    """Berechnet die Global Correlation (GCOR) innerhalb der Gehirnmaske."""
    voxels = data_4d[mask]
    means = np.mean(voxels, axis=-1, keepdims=True)
    stds = np.std(voxels, axis=-1, keepdims=True)
    stds[stds == 0] = 1.0

    z_voxels = (voxels - means) / stds
    global_signal = np.mean(z_voxels, axis=0)
    gcor = np.mean(global_signal ** 2)
    return gcor


def fit_dti_metrics(data_4d, bvals, bvecs, mask):
    """Berechnet FA und MD über OLS Tensor-Fitting für Histogramme."""
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T  # Formate anpassen

    # Design-Matrix X aufbauen
    X = np.zeros((len(bvals), 7))
    X[:, 0] = -bvals * bvecs[:, 0] ** 2
    X[:, 1] = -bvals * bvecs[:, 1] ** 2
    X[:, 2] = -bvals * bvecs[:, 2] ** 2
    X[:, 3] = -2 * bvals * bvecs[:, 0] * bvecs[:, 1]
    X[:, 4] = -2 * bvals * bvecs[:, 0] * bvecs[:, 2]
    X[:, 5] = -2 * bvals * bvecs[:, 1] * bvecs[:, 2]
    X[:, 6] = 1.0

    X_inv = np.linalg.pinv(X)

    # Daten extrahieren und Logarithmus berechnen
    voxels = data_4d[mask]
    voxels = np.clip(voxels, 1e-5, None)
    log_S = np.log(voxels).T

    # OLS Fitting
    beta = X_inv @ log_S

    # Tensor-Matrizen rekonstruieren
    num_voxels = beta.shape[1]
    D_matrices = np.zeros((num_voxels, 3, 3))
    D_matrices[:, 0, 0] = beta[0]
    D_matrices[:, 1, 1] = beta[1]
    D_matrices[:, 2, 2] = beta[2]
    D_matrices[:, 0, 1] = D_matrices[:, 1, 0] = beta[3]
    D_matrices[:, 0, 2] = D_matrices[:, 2, 0] = beta[4]
    D_matrices[:, 1, 2] = D_matrices[:, 2, 1] = beta[5]

    # Eigenwerte berechnen
    evals, _ = np.linalg.eigh(D_matrices)
    l1, l2, l3 = np.clip(evals[:, 2], 0, None), np.clip(evals[:, 1], 0, None), np.clip(evals[:, 0], 0, None)

    # MD (Mean Diffusivity)
    md = (l1 + l2 + l3) / 3.0

    # FA (Fractional Anisotropy)
    fa_num = np.sqrt((l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2)
    fa_den = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)

    fa = np.zeros(num_voxels)
    valid = fa_den > 0
    fa[valid] = np.sqrt(3 / 2) * (fa_num[valid] / fa_den[valid])
    fa = np.clip(fa, 0, 1)

    return fa, md


def visualize_comparison():
    # 1. Load the volumes
    noisy_nib = nib.load(NOISY_FILE)
    denoised_nib = nib.load(DENOISED_FILE)

    noisy_img = noisy_nib.get_fdata()
    denoised_img = denoised_nib.get_fdata()

    print(f"Loaded Noisy Image Shape:    {noisy_img.shape} ({noisy_img.ndim}D)")
    print(f"Loaded Denoised Image Shape: {denoised_img.shape} ({denoised_img.ndim}D)")

    # 2. Erzeuge repräsentative 3D-Strukturen
    noisy_mean_3d = np.mean(noisy_img, axis=-1) if noisy_img.ndim == 4 else noisy_img
    denoised_mean_3d = np.mean(denoised_img, axis=-1) if denoised_img.ndim == 4 else denoised_img

    brain_mask = create_brain_mask(noisy_mean_3d)
    is_denoised_4d = (denoised_img.ndim == 4)

    # 3. Metriken und DTI Parameter berechnen
    print("\n--- METRIKEN ERGEBNISSE ---")
    if noisy_img.ndim == 4:
        tsnr_noisy = calculate_tsnr(noisy_img)
        gcor_noisy = calculate_gcor(noisy_img, brain_mask)
        print(f"Global Correlation (GCOR) - Noisy:    {gcor_noisy:.4f}")
        print(f"Mittleres tSNR - Noisy:               {np.mean(tsnr_noisy[brain_mask]):.2f}")
    else:
        tsnr_noisy = None

    if is_denoised_4d:
        tsnr_denoised = calculate_tsnr(denoised_img)
        gcor_denoised = calculate_gcor(denoised_img, brain_mask)
        print(f"Global Correlation (GCOR) - Denoised: {gcor_denoised:.4f}")
        print(f"Mittleres tSNR - Denoised:            {np.mean(tsnr_denoised[brain_mask]):.2f}")
    print("---------------------------\n")

    # --- PLOT 1: Slice-by-Slice Residual Check ---
    z_mid = SLICE_IDX if SLICE_IDX is not None else denoised_mean_3d.shape[2] // 2
    noisy_slice = noisy_img[:, :, z_mid, 0] if noisy_img.ndim == 4 else noisy_img[:, :, z_mid]
    denoised_slice = denoised_img[:, :, z_mid, 0] if denoised_img.ndim == 4 else denoised_img[:, :, z_mid]
    difference = noisy_slice - denoised_slice

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    axes1[0].imshow(noisy_slice, cmap='gray')
    axes1[0].set_title("Noisy Input Slice")
    axes1[0].axis('off')
    axes1[1].imshow(denoised_slice, cmap='gray')
    axes1[1].set_title(f"Denoised Result ({'4D' if is_denoised_4d else '3D'})")
    axes1[1].axis('off')
    vmax = np.percentile(np.abs(difference), 99)
    axes1[2].imshow(difference, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes1[2].set_title("Removed Noise (Residual)")
    axes1[2].axis('off')
    plt.suptitle("Slice-by-Slice Residual Check", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- PLOT 2: Orthogonal Views ---
    x_mid, y_mid = denoised_mean_3d.shape[0] // 2, denoised_mean_3d.shape[1] // 2
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    axes2[0].imshow(np.rot90(denoised_mean_3d[:, :, z_mid]), cmap='gray')
    axes2[0].axis('off')
    axes2[1].imshow(np.rot90(denoised_mean_3d[:, y_mid, :]), cmap='gray')
    axes2[1].axis('off')
    axes2[2].imshow(np.rot90(denoised_mean_3d[x_mid, :, :]), cmap='gray')
    axes2[2].axis('off')
    plt.suptitle("Orthogonal Views of Denoised Volume", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- PLOT 3: 2x2 HISTOGRAM DASHBOARD ---
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Signal Intensity Histogram (Mean 3D)
    noisy_vals_hist = noisy_mean_3d[brain_mask]
    denoised_vals_hist = denoised_mean_3d[brain_mask]
    lb_sig, ub_sig = np.percentile(noisy_vals_hist, 1), np.percentile(noisy_vals_hist, 99)

    axes4[0, 0].hist(noisy_vals_hist, bins=100, range=(lb_sig, ub_sig), alpha=0.5, label='Noisy Input', color='red')
    axes4[0, 0].hist(denoised_vals_hist, bins=100, range=(lb_sig, ub_sig), alpha=0.5, label='Denoised', color='blue')
    axes4[0, 0].set_title("Signal Intensity (Mean Volume)")
    axes4[0, 0].legend()

    # 2. Residual Histogram (Full 4D Difference within mask)
    residuals_4d = (noisy_img - denoised_img)[brain_mask]
    lb_res, ub_res = np.percentile(residuals_4d, 0.5), np.percentile(residuals_4d, 99.5)

    axes4[0, 1].hist(residuals_4d.flatten(), bins=100, range=(lb_res, ub_res), alpha=0.7, color='purple')
    axes4[0, 1].axvline(0, color='black', linestyle='dashed', linewidth=1)
    axes4[0, 1].set_title("Residuals (Removed Noise Distribution)")

    # 3 & 4. FA and MD Distributions (Requires 4D data)
    if noisy_img.ndim == 4 and is_denoised_4d:
        print("Berechne FA und MD Verteilungen für Histogramme...")
        bvals = np.loadtxt(BVAL_FILE)
        bvecs = np.loadtxt(BVEC_FILE)

        fa_noisy, md_noisy = fit_dti_metrics(noisy_img, bvals, bvecs, brain_mask)
        fa_denoised, md_denoised = fit_dti_metrics(denoised_img, bvals, bvecs, brain_mask)

        # FA Plot
        axes4[1, 0].hist(fa_noisy, bins=80, range=(0, 1), alpha=0.5, label='Noisy FA', color='red')
        axes4[1, 0].hist(fa_denoised, bins=80, range=(0, 1), alpha=0.5, label='Denoised FA', color='blue')
        axes4[1, 0].set_title("Fractional Anisotropy (FA) Distribution")
        axes4[1, 0].legend()

        # MD Plot (clip extreme outliers for a clean plot)
        md_noisy = md_noisy[~np.isnan(md_noisy)]
        md_denoised = md_denoised[~np.isnan(md_denoised)]
        ub_md = np.percentile(md_noisy, 98)

        axes4[1, 1].hist(md_noisy, bins=80, range=(0, ub_md), alpha=0.5, label='Noisy MD', color='red')
        axes4[1, 1].hist(md_denoised, bins=80, range=(0, ub_md), alpha=0.5, label='Denoised MD', color='blue')
        axes4[1, 1].set_title("Mean Diffusivity (MD) Distribution")
        axes4[1, 1].legend()
    else:
        axes4[1, 0].text(0.5, 0.5, "Requires 4D Volumes for FA", ha='center')
        axes4[1, 1].text(0.5, 0.5, "Requires 4D Volumes for MD", ha='center')

    plt.suptitle("Statistical Evaluation (Within Brain Mask)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_comparison()