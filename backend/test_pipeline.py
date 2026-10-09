import os
import sys
import numpy as np

# ------------------------------------------------------------
# Add backend directory to Python path
# ------------------------------------------------------------

BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BACKEND_DIR
)

sys.path.insert(0, BACKEND_DIR)

from inference.pipeline import BrainTumorEdgePipeline


# ------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------

SEGMENTATION_MODEL = os.path.join(
    BACKEND_DIR,
    "models",
    "unet_brain_tumor_int8.onnx"
)

CLASSIFICATION_MODEL = os.path.join(
    BACKEND_DIR,
    "models",
    "mobilenetv3_small_brain_tumor.onnx"
)


# ------------------------------------------------------------
# PATIENT DATA
# ------------------------------------------------------------

# Change this path to your local BraTS NPZ file.
#
# Example:
# D:\Major Project\data\BraTS2021_00019.npz

PATIENT_NPZ = os.path.join(
    PROJECT_DIR,
    "data",
    "BraTS2021_00019.npz"
)


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("=" * 70)
print("BRAIN TUMOR EDGE AI - PIPELINE TEST")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Segmentation model:")
print(SEGMENTATION_MODEL)

print()
print("Classification model:")
print(CLASSIFICATION_MODEL)

print()
print("Patient file:")
print(PATIENT_NPZ)


required_files = [
    SEGMENTATION_MODEL,
    CLASSIFICATION_MODEL,
    PATIENT_NPZ
]

for path in required_files:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

print()
print("✓ All required files found")


# ------------------------------------------------------------
# LOAD PATIENT
# ------------------------------------------------------------

print()
print("=" * 70)
print("LOADING PATIENT")
print("=" * 70)

data = np.load(PATIENT_NPZ)

print()
print("NPZ keys:")
print(data.files)

if "images" not in data:
    raise KeyError(
        "NPZ file does not contain 'images'."
    )

volume = data["images"].astype(
    np.float32
)

print()
print("Volume shape:", volume.shape)
print("Volume dtype:", volume.dtype)


# ------------------------------------------------------------
# VALIDATE VOLUME
# ------------------------------------------------------------

expected_shape = (
    4,
    240,
    240,
    155
)

if volume.shape != expected_shape:

    raise ValueError(
        f"\nExpected volume shape "
        f"{expected_shape}, "
        f"got {volume.shape}"
    )

print()
print("✓ Volume shape is correct")


# ------------------------------------------------------------
# CREATE PIPELINE
# ------------------------------------------------------------

print()
print("=" * 70)
print("CREATING EDGE AI PIPELINE")
print("=" * 70)

pipeline = BrainTumorEdgePipeline(
    segmentation_model_path=SEGMENTATION_MODEL,
    classification_model_path=CLASSIFICATION_MODEL
)


# ------------------------------------------------------------
# RUN COMPLETE PIPELINE
# ------------------------------------------------------------

print()
print("=" * 70)
print("RUNNING COMPLETE EDGE AI PIPELINE")
print("=" * 70)

result = pipeline.analyze(
    volume=volume,
    case_id="BraTS2021_00019"
)


# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL PIPELINE RESULT")
print("=" * 70)

print()

print(
    "Case ID:",
    result["case_id"]
)

print(
    "Tumor detected:",
    result["tumor_detected"]
)

print(
    "Tumor slices:",
    result["tumor_slice_count"]
)

print(
    "Selected slices:",
    result["selected_slices"]
)

print()

print("Class probabilities:")

for class_name, probability in result[
    "probabilities"
].items():

    print(
        f"  {class_name:12s}: "
        f"{probability * 100:.4f}%"
    )

print()

print(
    "Prediction:",
    result["prediction"]
)

print(
    "Confidence:",
    f"{result['confidence'] * 100:.2f}%"
)

print(
    "Entropy:",
    f"{result['entropy']:.4f}"
)

print(
    "Normalized entropy:",
    f"{result['normalized_entropy']:.4f}"
)

print(
    "Top-1 / Top-2 margin:",
    f"{result['margin']:.4f}"
)

print(
    "Uncertainty status:",
    result["uncertainty_status"]
)

print()

print("Recommendation:")
print(result["recommendation"])


# ------------------------------------------------------------
# FINAL CHECKS
# ------------------------------------------------------------

assert "predicted_mask" in result

assert result[
    "predicted_mask"
].shape == (
    240,
    240,
    155
)

assert isinstance(
    result["probabilities"],
    dict
)

assert len(
    result["probabilities"]
) == 4

print()
print("=" * 70)
print("✓ COMPLETE EDGE AI PIPELINE TEST PASSED")
print("=" * 70)