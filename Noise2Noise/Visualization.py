import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# --- Settings ---
NOISY_FILE = "../data/1_high_res.nii.gz"
DENOISED_FILE = "result.nii.gz"
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


def visualize_comparison():
    # 1. Load the volumes
    noisy_nib = nib.load(NOISY_FILE)
    denoised_nib = nib.load(DENOISED_FILE)

    noisy_img = noisy_nib.get_fdata()
    denoised_img = denoised_nib.get_fdata()

    print(f"Loaded Noisy Image Shape:    {noisy_img.shape} ({noisy_img.ndim}D)")
    print(f"Loaded Denoised Image Shape: {denoised_img.shape} ({denoised_img.ndim}D)")

    # 2. Erzeuge repräsentative 3D-Strukturen für die Maskierung und Anzeige
    noisy_mean_3d = np.mean(noisy_img, axis=-1) if noisy_img.ndim == 4 else noisy_img
    denoised_mean_3d = np.mean(denoised_img, axis=-1) if denoised_img.ndim == 4 else denoised_img

    # Gehirnmaske auf Basis des Noisy-Bildes berechnen
    brain_mask = create_brain_mask(noisy_mean_3d)

    # 3. Metriken berechnen (Dynamisch je nach Dimension)
    print("\n--- METRIKEN ERGEBNISSE ---")

    # Noisy Metriken (erwartet 4D)
    if noisy_img.ndim == 4:
        tsnr_noisy = calculate_tsnr(noisy_img)
        gcor_noisy = calculate_gcor(noisy_img, brain_mask)
        print(f"Global Correlation (GCOR) - Noisy:    {gcor_noisy:.4f}")
        print(f"Mittleres tSNR - Noisy:               {np.mean(tsnr_noisy[brain_mask]):.2f}")
    else:
        print("Noisy Image ist nicht 4D. Temporale Metriken für Noisy übersprungen.")
        tsnr_noisy = None

    # Denoised Metriken (nur wenn die entrauschte Datei ebenfalls 4D ist)
    is_denoised_4d = (denoised_img.ndim == 4)
    if is_denoised_4d:
        tsnr_denoised = calculate_tsnr(denoised_img)
        gcor_denoised = calculate_gcor(denoised_img, brain_mask)
        print(f"Global Correlation (GCOR) - Denoised: {gcor_denoised:.4f}")
        print(f"Mittleres tSNR - Denoised:            {np.mean(tsnr_denoised[brain_mask]):.2f}")
    else:
        print("Hinweis: Denoised Image ist 3D. Temporale Metriken (tSNR/GCOR) für Denoised übersprungen.")
        print("-> Um tSNR für Denoised zu berechnen, musst du die gesamte 4D-Serie entrauschen und übergeben.")
    print("---------------------------\n")

    # Indizes für die Visualisierung bestimmen (Mitte des Volumens)
    x_mid = denoised_mean_3d.shape[0] // 2
    y_mid = denoised_mean_3d.shape[1] // 2
    z_mid = SLICE_IDX if SLICE_IDX is not None else denoised_mean_3d.shape[2] // 2

    # --- PLOT 1: Klassischer Residual Check (Erster verfügbarer Frame) ---
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

    # --- PLOT 2: Orthogonale Ansichten (Struktur-Check) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    axes2[0].imshow(np.rot90(denoised_mean_3d[:, :, z_mid]), cmap='gray')
    axes2[0].set_title(f"Axial View (Slice {z_mid})")
    axes2[0].axis('off')

    axes2[1].imshow(np.rot90(denoised_mean_3d[:, y_mid, :]), cmap='gray')
    axes2[1].set_title(f"Coronal View (Slice {y_mid})")
    axes2[1].axis('off')

    axes2[2].imshow(np.rot90(denoised_mean_3d[x_mid, :, :]), cmap='gray')
    axes2[2].set_title(f"Sagittal View (Slice {x_mid})")
    axes2[2].axis('off')
    plt.suptitle("Orthogonal Views of Denoised Volume (Structural Check)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # --- PLOT 3: tSNR Map Vergleich (nur wenn beide 4D sind) ---
    if tsnr_noisy is not None and is_denoised_4d:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        vmax_tsnr = np.percentile(tsnr_denoised[brain_mask], 98)

        im_a = axes3[0].imshow(tsnr_noisy[:, :, z_mid], cmap='viridis', vmin=0, vmax=vmax_tsnr)
        axes3[0].set_title("tSNR Map - Noisy")
        axes3[0].axis('off')
        fig3.colorbar(im_a, ax=axes3[0], fraction=0.046, pad=0.04)

        im_b = axes3[1].imshow(tsnr_denoised[:, :, z_mid], cmap='viridis', vmin=0, vmax=vmax_tsnr)
        axes3[1].set_title("tSNR Map - Denoised")
        axes3[1].axis('off')
        fig3.colorbar(im_b, ax=axes3[1], fraction=0.046, pad=0.04)
        plt.suptitle("Temporal SNR Maps (Comparison)", fontsize=14, fontweight='bold')
        plt.tight_layout()
    else:
        print("Plot 3 (tSNR Maps) übersprungen, da das Denoised-Bild keine 4D-Zeitreihe ist.")

    # --- PLOT 4: Signal-Histogramme innerhalb der Gehirnmaske ---
    plt.figure(figsize=(10, 5))

    # Wenn 4D, nehmen wir den Mittelwert über die Zeit für das Histogramm, um 3D-Strukturen zu vergleichen
    noisy_vals_hist = noisy_mean_3d[brain_mask]
    denoised_vals_hist = denoised_mean_3d[brain_mask]

    lower_b = np.percentile(noisy_vals_hist, 1)
    upper_b = np.percentile(noisy_vals_hist, 99)

    plt.hist(noisy_vals_hist, bins=100, range=(lower_b, upper_b), alpha=0.5, label='Noisy Input (Mean)', color='red')
    plt.hist(denoised_vals_hist, bins=100, range=(lower_b, upper_b), alpha=0.5, label='Denoised Output', color='blue')

    plt.title("Signal Intensity Histogram (Within Brain Mask Only)", fontsize=14, fontweight='bold')
    plt.xlabel("Voxel Intensity")
    plt.ylabel("Voxel Count")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    visualize_comparison()