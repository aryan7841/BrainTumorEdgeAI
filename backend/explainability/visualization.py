import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_tumor_explanation(
    volume,
    predicted_mask,
    selected_slices,
    output_dir,
    case_id="unknown"
):
    """
    Creates a segmentation-based explainability image.

    The visualization shows:
    1. MRI image
    2. U-Net predicted tumor mask
    3. MRI + tumor overlay
    4. Tumor ROI / bounding box

    Returns:
        Path to the generated PNG image.
    """

    os.makedirs(output_dir, exist_ok=True)

    if volume is None:
        raise ValueError("MRI volume is None.")

    if predicted_mask is None:
        raise ValueError("Predicted mask is None.")

    volume = np.asarray(volume)
    predicted_mask = np.asarray(predicted_mask)

    if volume.ndim != 4:
        raise ValueError(
            f"Expected volume shape (4,H,W,Z), got {volume.shape}"
        )

    if predicted_mask.ndim != 3:
        raise ValueError(
            f"Expected predicted mask shape (H,W,Z), got {predicted_mask.shape}"
        )

    if len(selected_slices) == 0:
        raise ValueError("No selected tumor slices available.")

    # Use the first selected tumor slice.
    slice_idx = int(selected_slices[0])

    if slice_idx < 0 or slice_idx >= volume.shape[3]:
        raise ValueError(
            f"Invalid slice index {slice_idx} for volume depth {volume.shape[3]}"
        )

    # ---------------------------------------------------------
    # Create grayscale MRI from the four modalities
    # ---------------------------------------------------------

    mri_slice = volume[:, :, :, slice_idx]

    # Average the 4 modalities.
    mri_image = np.mean(mri_slice, axis=0)

    # Normalize only for visualization.
    mri_min = np.min(mri_image)
    mri_max = np.max(mri_image)

    if mri_max > mri_min:
        mri_image = (
            mri_image - mri_min
        ) / (mri_max - mri_min)
    else:
        mri_image = np.zeros_like(mri_image)

    # ---------------------------------------------------------
    # Get tumor mask for selected slice
    # ---------------------------------------------------------

    tumor_mask = predicted_mask[:, :, slice_idx] > 0

    # ---------------------------------------------------------
    # Calculate tumor bounding box
    # ---------------------------------------------------------

    ys, xs = np.where(tumor_mask)

    bbox = None

    if len(xs) > 0 and len(ys) > 0:
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        bbox = (
            x_min,
            y_min,
            x_max - x_min + 1,
            y_max - y_min + 1
        )

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    # ---------------------------------------------------------
    # 1. Original MRI
    # ---------------------------------------------------------

    axes[0].imshow(
        mri_image,
        cmap="gray"
    )

    axes[0].set_title(
        f"MRI Slice {slice_idx}"
    )

    axes[0].axis("off")

    # ---------------------------------------------------------
    # 2. Predicted tumor mask
    # ---------------------------------------------------------

    axes[1].imshow(
        tumor_mask,
        cmap="gray"
    )

    axes[1].set_title(
        "U-Net Tumor Mask"
    )

    axes[1].axis("off")

    # ---------------------------------------------------------
    # 3. MRI + tumor overlay
    # ---------------------------------------------------------

    axes[2].imshow(
        mri_image,
        cmap="gray"
    )

    overlay = np.ma.masked_where(
        ~tumor_mask,
        tumor_mask
    )

    axes[2].imshow(
        overlay,
        cmap="autumn",
        alpha=0.55
    )

    axes[2].set_title(
        "Tumor Localization"
    )

    axes[2].axis("off")

    # ---------------------------------------------------------
    # 4. ROI / Bounding Box
    # ---------------------------------------------------------

    axes[3].imshow(
        mri_image,
        cmap="gray"
    )

    if bbox is not None:
        x, y, width, height = bbox

        rectangle = Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            linewidth=2
        )

        axes[3].add_patch(rectangle)

        axes[3].set_title(
            "Detected Tumor ROI"
        )

    else:
        axes[3].set_title(
            "No Tumor ROI"
        )

    axes[3].axis("off")

    # ---------------------------------------------------------
    # Overall title
    # ---------------------------------------------------------

    fig.suptitle(
        f"AI Explainability - {case_id}",
        fontsize=14
    )

    plt.tight_layout()

    # ---------------------------------------------------------
    # Save image
    # ---------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        f"{case_id}_tumor_explainability.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path