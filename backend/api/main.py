import os
import tempfile
import shutil

import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.inference.pipeline import BrainTumorEdgePipeline
from backend.reporting.report_generator import BrainTumorReportGenerator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

EXPLAINABILITY_DIR = os.path.join(
    REPORT_DIR,
    "explainability"
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    EXPLAINABILITY_DIR,
    exist_ok=True
)


# ============================================================
# MODEL PATHS
# ============================================================

SEGMENTATION_MODEL = os.path.join(
    MODEL_DIR,
    "unet_brain_tumor_int8.onnx"
)

CLASSIFICATION_MODEL = os.path.join(
    MODEL_DIR,
    "mobilenetv3_small_brain_tumor.onnx"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="BrainTumorEdgeAI",
    description=(
        "Edge AI-based brain MRI segmentation, "
        "classification, uncertainty estimation, "
        "explainability and AI-assisted decision support."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIALIZE PIPELINE
# ============================================================

print()
print("=" * 70)
print("STARTING BRAIN TUMOR EDGE AI BACKEND")
print("=" * 70)

print()
print("Segmentation model:")
print(SEGMENTATION_MODEL)

print()
print("Classification model:")
print(CLASSIFICATION_MODEL)

if not os.path.exists(SEGMENTATION_MODEL):
    raise FileNotFoundError(
        "Segmentation model not found:\n"
        f"{SEGMENTATION_MODEL}"
    )

if not os.path.exists(CLASSIFICATION_MODEL):
    raise FileNotFoundError(
        "Classification model not found:\n"
        f"{CLASSIFICATION_MODEL}"
    )


pipeline = BrainTumorEdgePipeline(
    segmentation_model_path=SEGMENTATION_MODEL,
    classification_model_path=CLASSIFICATION_MODEL
)

report_generator = BrainTumorReportGenerator(output_dir=REPORT_DIR)

print()
print("✓ Backend initialization completed")
print("=" * 70)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "project": "BrainTumorEdgeAI",
        "status": "running",
        "description": (
            "AI-assisted brain MRI analysis "
            "using local Edge AI inference."
        )
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pipeline": "ready"
    }


@app.get("/model-info")
def model_info():
    return {
        "segmentation": {
            "model": "Lightweight U-Net",
            "runtime": "ONNX Runtime",
            "optimization": "INT8",
            "file": os.path.basename(
                SEGMENTATION_MODEL
            )
        },
        "classification": {
            "model": "MobileNetV3-Small",
            "runtime": "ONNX Runtime",
            "classes": [
                "glioma",
                "meningioma",
                "no_tumor",
                "pituitary"
            ],
            "file": os.path.basename(
                CLASSIFICATION_MODEL
            )
        },
        "explainability": {
            "method": (
                "Segmentation-based tumor "
                "localization visualization"
            ),
            "gradcam": False
        }
    }


# ============================================================
# REPORT PATH HELPER
# ============================================================

def get_report_path(report_result):
    """
    Handle both possible report-generator return schemas.

    Supported:
        {"pdf_path": "..."}
        {"pdf": "..."}

    Returns:
        PDF path
    """

    if not report_result:
        return None

    pdf_path = report_result.get(
        "pdf_path"
    )

    if pdf_path is None:
        pdf_path = report_result.get(
            "pdf"
        )

    return pdf_path


# ============================================================
# SAFE NPZ LOADING
# ============================================================

def load_mri_npz(file_path):
    """
    Load MRI volume from NPZ.

    Expected shape:
        (4, 240, 240, 155)
    """

    try:

        with np.load(
            file_path,
            allow_pickle=False
        ) as data:

            if "images" not in data:
                raise ValueError(
                    "NPZ file does not contain "
                    "an 'images' array."
                )

            volume = np.array(
                data["images"],
                copy=True
            )

    except Exception as exc:

        raise ValueError(
            f"Could not read NPZ file: {exc}"
        )

    if volume.shape != (
        4,
        240,
        240,
        155
    ):
        raise ValueError(
            "Invalid MRI volume shape. "
            "Expected (4, 240, 240, 155), "
            f"got {volume.shape}"
        )

    if not np.issubdtype(
        volume.dtype,
        np.number
    ):
        raise ValueError(
            "MRI volume must contain "
            "numeric values."
        )

    if not np.isfinite(
        volume
    ).all():
        raise ValueError(
            "MRI volume contains NaN "
            "or infinite values."
        )

    return volume.astype(
        np.float32
    )


# ============================================================
# PREPARE REPORT DATA
# ============================================================

def prepare_report_data(
    result,
    case_id
):
    """
    Convert pipeline result into the schema
    expected by the Notebook-6-style report.
    """

    probabilities = result.get(
        "probabilities",
        {}
    )

    prediction = result.get(
        "prediction"
    )

    # Determine highest probability class
    if probabilities:

        valid_probabilities = {
            key: float(value)
            for key, value
            in probabilities.items()
        }

        highest_probability_class = max(
            valid_probabilities,
            key=valid_probabilities.get
        )

    else:

        valid_probabilities = {}

        highest_probability_class = (
            prediction
            if prediction
            else "N/A"
        )

    if prediction is None:
        prediction_for_report = "N/A"
    else:
        prediction_for_report = prediction

    report_data = {

        # ----------------------------------------------------
        # Case information
        # ----------------------------------------------------

        "case_id":
            case_id,

        "report_title":
            "BrainTumorEdgeAI AI-Assisted MRI Report",

        # ----------------------------------------------------
        # MRI information
        # ----------------------------------------------------

        "modalities": [
            "T1",
            "T1CE",
            "T2",
            "FLAIR"
        ],

        "images_shape":
            result.get(
                "images_shape",
                ()
            ),

        "segmentation_model":
            "Lightweight U-Net",

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        "tumor_detected":
            bool(
                result.get(
                    "tumor_detected",
                    False
                )
            ),

        "tumor_slice_count":
            int(
                result.get(
                    "tumor_slices",
                    result.get(
                        "tumor_slice_count",
                        0
                    )
                )
            ),

        "selected_slices":
            result.get(
                "selected_slices",
                []
            ),

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        "prediction":
            prediction_for_report,

        "highest_probability_class":
            highest_probability_class,

        "confidence":
            float(
                result.get(
                    "confidence",
                    0.0
                )
            ),

        "probabilities":
            valid_probabilities,

        "classification_model":
            "MobileNetV3-Small",

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        "entropy":
            float(
                result.get(
                    "entropy",
                    0.0
                )
            ),

        "normalized_entropy":
            float(
                result.get(
                    "normalized_entropy",
                    0.0
                )
            ),

        "margin":
            float(
                result.get(
                    "top2_margin",
                    result.get(
                        "margin",
                        0.0
                    )
                )
            ),

        "top2_margin":
            float(
                result.get(
                    "top2_margin",
                    result.get(
                        "margin",
                        0.0
                    )
                )
            ),

        "uncertainty_status":
            result.get(
                "uncertainty_status",
                "UNCERTAIN / REVIEW"
            ),

        # ----------------------------------------------------
        # Decision support
        # ----------------------------------------------------

        "recommendation":
            result.get(
                "guidance",
                result.get(
                    "recommendation",
                    ""
                )
            ),

        "guidance":
            result.get(
                "guidance",
                result.get(
                    "recommendation",
                    ""
                )
            ),

        # ----------------------------------------------------
        # Explainability
        # ----------------------------------------------------

        "explainability_image":
            result.get(
                "explainability_image"
            ),

        "gradcam_available":
            False,

        # ----------------------------------------------------
        # Important project positioning
        # ----------------------------------------------------

        "is_clinical_diagnosis":
            False,

        "clinical_disclaimer":
            (
                "This system is an AI-assisted research "
                "prototype for decision support. It is not "
                "a clinically validated autonomous diagnostic "
                "system and does not replace a qualified "
                "radiologist or physician."
            )
    }

    return report_data


# ============================================================
# TEST ENDPOINT
# ============================================================

@app.get("/analyze-test")
def analyze_test():
    """
    Simple endpoint used to verify that the API and
    pipeline are initialized correctly.

    It does not run MRI inference.
    """

    return {
        "status": "ready",
        "message": (
            "Upload a valid BraTS2021 NPZ file "
            "to /analyze."
        ),
        "expected_shape": [
            4,
            240,
            240,
            155
        ]
    }


# ============================================================
# ANALYZE MRI
# ============================================================

@app.post("/analyze")
async def analyze_mri(
    file: UploadFile = File(...)
):
    """
    Run complete Edge AI analysis.

    Workflow:

        NPZ upload
            ↓
        validation
            ↓
        U-Net segmentation
            ↓
        tumor slice selection
            ↓
        ROI extraction
            ↓
        MobileNet classification
            ↓
        uncertainty
            ↓
        explainability image
            ↓
        PDF report
            ↓
        JSON response
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied."
        )

    if not file.filename.lower().endswith(
        ".npz"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. "
                "Please upload an NPZ MRI file."
            )
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Case ID
        # ----------------------------------------------------

        original_filename = os.path.basename(
            file.filename
        )

        case_id = os.path.splitext(
            original_filename
        )[0]

        # ----------------------------------------------------
        # Temporary upload
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".npz",
            dir=UPLOAD_DIR
        )

        temp_path = temp_file.name

        try:

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        finally:

            temp_file.close()

        print()
        print("=" * 70)
        print("MRI UPLOAD RECEIVED")
        print("=" * 70)

        print(
            "Filename:",
            original_filename
        )

        print(
            "Case ID:",
            case_id
        )

        print(
            "Temporary file:",
            temp_path
        )

        # ----------------------------------------------------
        # Load volume
        # ----------------------------------------------------

        volume = load_mri_npz(
            temp_path
        )

        print()
        print(
            "MRI volume shape:",
            volume.shape
        )

        print(
            "MRI dtype:",
            volume.dtype
        )

        # ----------------------------------------------------
        # Run pipeline
        # ----------------------------------------------------

        result = pipeline.analyze(
            volume,
            case_id=case_id
        )

        # ----------------------------------------------------
        # Prepare report
        # ----------------------------------------------------

        report_data = prepare_report_data(
            result,
            case_id
        )

        explainability_image = result.get(
            "explainability_image"
        )

        # ----------------------------------------------------
        # Generate PDF
        # ----------------------------------------------------

        report_result = (
            report_generator.generate(
                report_data=report_data,
                pipeline_image_path=(
                    explainability_image
                ),
                uncertainty_image_path=None,
                gradcam_image_path=None
            )
        )

        pdf_path = get_report_path(
            report_result
        )

        if not pdf_path:
            raise RuntimeError(
                "Report generator did not "
                "return a PDF path."
            )

        if not os.path.exists(
            pdf_path
        ):
            raise RuntimeError(
                "PDF report was not created:\n"
                f"{pdf_path}"
            )

        # ----------------------------------------------------
        # API-safe result
        # ----------------------------------------------------

        api_result = dict(
            result
        )

        # Never send the complete 3D mask through JSON.
        api_result.pop(
            "predicted_mask",
            None
        )

        # Add report information
        api_result[
            "report"
        ] = {
            "pdf_path":
                pdf_path,

            "pdf_filename":
                os.path.basename(
                    pdf_path
                )
        }

        # Explicit explainability information
        api_result[
            "explainability"
        ] = {
            "available":
                bool(
                    explainability_image
                    and
                    os.path.exists(
                        explainability_image
                    )
                ),

            "image_path":
                explainability_image,

            "method":
                (
                    "Segmentation-based tumor "
                    "localization visualization"
                ),

            "gradcam_available":
                False
        }

        print()
        print("=" * 70)
        print("API ANALYSIS COMPLETED")
        print("=" * 70)

        print(
            "Case ID:",
            case_id
        )

        print(
            "Tumor slices:",
            api_result.get(
                "tumor_slices",
                api_result.get(
                    "tumor_slice_count",
                    0
                )
            )
        )

        print(
            "Prediction:",
            api_result.get(
                "prediction"
            )
        )

        print(
            "Confidence:",
            f"{api_result.get('confidence', 0) * 100:.2f}%"
        )

        print(
            "Explainability image:",
            explainability_image
        )

        print(
            "PDF:",
            pdf_path
        )

        print("=" * 70)

        return JSONResponse(
            content=api_result
        )

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print("ANALYSIS ERROR")
        print("=" * 70)

        print(
            type(exc).__name__,
            ":",
            str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "MRI analysis failed: "
                f"{type(exc).__name__}: {exc}"
            )
        )

    finally:

        # ----------------------------------------------------
        # Clean temporary upload
        # ----------------------------------------------------

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass


# ============================================================
# REPORT DOWNLOAD
# ============================================================

@app.get(
    "/reports/{report_name}"
)
def download_report(
    report_name: str
):
    """
    Download a generated PDF report.
    """

    # Security: only use the filename, never a path.
    safe_name = os.path.basename(
        report_name
    )

    if safe_name != report_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid report filename."
        )

    report_path = os.path.join(
        REPORT_DIR,
        safe_name
    )

    if not os.path.exists(
        report_path
    ):
        raise HTTPException(
            status_code=404,
            detail="Report not found."
        )

    if not safe_name.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF reports can be downloaded."
        )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=safe_name
    )


# ============================================================
# EXPLAINABILITY IMAGE
# ============================================================

@app.get(
    "/explainability/{image_name}"
)
def get_explainability_image(
    image_name: str
):
    """
    Return generated tumor explainability PNG.
    """

    safe_name = os.path.basename(
        image_name
    )

    if safe_name != image_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid image filename."
        )

    image_path = os.path.join(
        EXPLAINABILITY_DIR,
        safe_name
    )

    if not os.path.exists(
        image_path
    ):
        raise HTTPException(
            status_code=404,
            detail="Explainability image not found."
        )

    if not safe_name.lower().endswith(
        ".png"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PNG explainability images are supported."
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=safe_name
    )