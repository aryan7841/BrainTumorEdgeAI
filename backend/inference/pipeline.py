import os
import numpy as np

from .segmentation import BrainTumorSegmenter
from .classification import BrainTumorClassifier

from backend.explainability.visualization import create_tumor_explanation

class BrainTumorEdgePipeline:
    """
    Complete Edge AI inference pipeline.

    MRI volume
        ↓
    INT8 ONNX U-Net segmentation
        ↓
    Tumor slice selection
        ↓
    ROI extraction
        ↓
    ONNX MobileNetV3 classification
        ↓
    Multi-slice probability aggregation
        ↓
    Uncertainty estimation
        ↓
    Explainability visualization
        ↓
    Clinical decision-support guidance
    """

    CLASS_NAMES = [
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary"
    ]

    # Prototype thresholds.
    # These are NOT clinically validated.
    CONFIDENCE_THRESHOLD = 0.60
    ENTROPY_THRESHOLD = 0.70
    MARGIN_THRESHOLD = 0.20

    def __init__(
        self,
        segmentation_model_path: str,
        classification_model_path: str
    ):
        print("=" * 70)
        print("INITIALIZING BRAIN TUMOR EDGE AI PIPELINE")
        print("=" * 70)

        self.segmenter = BrainTumorSegmenter(
            segmentation_model_path
        )

        self.classifier = BrainTumorClassifier(
            classification_model_path
        )

        print()
        print("✓ Edge AI pipeline initialized")
        print("✓ Segmentation: INT8 ONNX U-Net")
        print("✓ Classification: ONNX MobileNetV3-Small")
        print("✓ Explainability: U-Net tumor localization visualization")

    # ==========================================================
    # UNCERTAINTY
    # ==========================================================

    def calculate_uncertainty(
        self,
        probabilities: np.ndarray
    ):
        """
        Calculate:

        - confidence
        - entropy
        - normalized entropy
        - top-1/top-2 probability margin

        Same mathematical logic used previously.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64
        )

        probabilities = np.clip(
            probabilities,
            1e-10,
            1.0
        )

        probabilities = (
            probabilities /
            probabilities.sum()
        )

        sorted_probs = np.sort(
            probabilities
        )[::-1]

        confidence = float(
            sorted_probs[0]
        )

        margin = float(
            sorted_probs[0] -
            sorted_probs[1]
        )

        entropy = float(
            -np.sum(
                probabilities *
                np.log(probabilities)
            )
        )

        max_entropy = np.log(
            len(self.CLASS_NAMES)
        )

        normalized_entropy = float(
            entropy / max_entropy
        )

        return (
            confidence,
            entropy,
            normalized_entropy,
            margin
        )

    def determine_uncertainty_status(
        self,
        confidence,
        normalized_entropy,
        margin
    ):
        """
        Prototype uncertainty decision.

        These thresholds are NOT clinically validated.
        """

        uncertain = (
            confidence < self.CONFIDENCE_THRESHOLD
            or
            normalized_entropy > self.ENTROPY_THRESHOLD
            or
            margin < self.MARGIN_THRESHOLD
        )

        if uncertain:
            return "UNCERTAIN / REVIEW"

        return "CONFIDENT"

    # ==========================================================
    # TUMOR SLICE SELECTION
    # ==========================================================

    def get_tumor_slices(
        self,
        predicted_mask: np.ndarray,
        min_pixels: int = 50
    ):
        """
        Find axial slices containing at least
        min_pixels tumor pixels.
        """

        if predicted_mask.ndim != 3:
            raise ValueError(
                "Expected predicted mask shape "
                "(240, 240, 155), got "
                f"{predicted_mask.shape}"
            )

        tumor_pixels_per_slice = np.sum(
            predicted_mask > 0,
            axis=(0, 1)
        )

        tumor_slices = np.where(
            tumor_pixels_per_slice >= min_pixels
        )[0]

        return (
            tumor_slices,
            tumor_pixels_per_slice
        )

    def select_top_tumor_slices(
        self,
        tumor_slices,
        tumor_pixels_per_slice,
        top_k: int = 7
    ):
        """
        Select slices containing the largest
        tumor regions.
        """

        if len(tumor_slices) == 0:
            return []

        sorted_slices = sorted(
            tumor_slices,
            key=lambda s: tumor_pixels_per_slice[s],
            reverse=True
        )

        return [
            int(s)
            for s in sorted_slices[:top_k]
        ]

    # ==========================================================
    # ROI EXTRACTION
    # ==========================================================

    def extract_tumor_roi(
        self,
        volume: np.ndarray,
        predicted_mask: np.ndarray,
        slice_idx: int,
        margin: int = 20
    ):
        """
        Extract a tumor-centered 4-modality ROI.

        Returns:
            roi
            roi_mask
            bbox
        """

        image_slice = volume[
            :, :, :, slice_idx
        ]

        mask_slice = predicted_mask[
            :, :, slice_idx
        ]

        tumor_coords = np.where(
            mask_slice > 0
        )

        if len(tumor_coords[0]) == 0:
            return None, None, None

        y_min = max(
            int(tumor_coords[0].min()) - margin,
            0
        )

        y_max = min(
            int(tumor_coords[0].max()) + margin + 1,
            image_slice.shape[1]
        )

        x_min = max(
            int(tumor_coords[1].min()) - margin,
            0
        )

        x_max = min(
            int(tumor_coords[1].max()) + margin + 1,
            image_slice.shape[2]
        )

        roi = image_slice[
            :,
            y_min:y_max,
            x_min:x_max
        ]

        roi_mask = mask_slice[
            y_min:y_max,
            x_min:x_max
        ]

        bbox = (
            y_min,
            x_min,
            y_max,
            x_max
        )

        return (
            roi,
            roi_mask,
            bbox
        )

    # ==========================================================
    # BRATS → MOBILENET ROI BRIDGE
    # ==========================================================

    def convert_roi_to_classifier_input(
        self,
        roi: np.ndarray
    ):
        """
        Convert 4-modality BraTS ROI to the 2D grayscale
        representation expected by the deployed MobileNet
        preprocessing.

        Current integration bridge:

        4 MRI modalities
                ↓
        mean grayscale
                ↓
        MobileNet preprocessing
        """

        if roi is None:
            raise ValueError("ROI is None.")

        if roi.ndim != 3:
            raise ValueError(
                "Expected ROI shape "
                "(4, H, W), got "
                f"{roi.shape}"
            )

        if roi.shape[0] != 4:
            raise ValueError(
                "Expected 4 MRI modalities, got "
                f"{roi.shape[0]}"
            )

        grayscale_roi = np.mean(
            roi.astype(np.float32),
            axis=0
        )

        return grayscale_roi

    # ==========================================================
    # CLASSIFICATION
    # ==========================================================

    def classify_selected_slices(
        self,
        volume: np.ndarray,
        predicted_mask: np.ndarray,
        selected_slices,
        roi_margin: int = 20
    ):
        """
        Classify each selected tumor ROI separately.

        The current MobileNet ONNX graph is used
        with batch size 1.
        """

        slice_results = []
        probability_list = []

        for slice_idx in selected_slices:

            roi, roi_mask, bbox = (
                self.extract_tumor_roi(
                    volume,
                    predicted_mask,
                    int(slice_idx),
                    margin=roi_margin
                )
            )

            if roi is None:
                continue

            classifier_roi = (
                self.convert_roi_to_classifier_input(
                    roi
                )
            )

            classification = (
                self.classifier.predict(
                    classifier_roi
                )
            )

            probabilities = np.array(
                [
                    classification["probabilities"]
                    [class_name]
                    for class_name
                    in self.CLASS_NAMES
                ],
                dtype=np.float32
            )

            probability_list.append(
                probabilities
            )

            slice_results.append({
                "slice": int(slice_idx),

                "prediction":
                    classification["prediction"],

                "confidence":
                    float(classification["confidence"]),

                "probabilities":
                    classification["probabilities"],

                "bbox": bbox,

                "roi_shape":
                    tuple(roi.shape)
            })

        if len(probability_list) == 0:
            raise RuntimeError(
                "No valid tumor ROIs were classified."
            )

        probability_matrix = np.stack(
            probability_list,
            axis=0
        )

        # Patient-level aggregation
        aggregated_probabilities = np.mean(
            probability_matrix,
            axis=0
        )

        aggregated_probabilities = (
            aggregated_probabilities /
            aggregated_probabilities.sum()
        )

        final_index = int(
            np.argmax(
                aggregated_probabilities
            )
        )

        final_prediction = (
            self.CLASS_NAMES[
                final_index
            ]
        )

        final_confidence = float(
            aggregated_probabilities[
                final_index
            ]
        )

        return (
            slice_results,
            aggregated_probabilities,
            final_prediction,
            final_confidence
        )

    # ==========================================================
    # DECISION SUPPORT
    # ==========================================================

    def generate_patient_guidance(
        self,
        tumor_detected,
        predicted_class,
        uncertainty_status,
        confidence,
        top2_margin
    ):
        """
        Conservative, non-diagnostic guidance.

        This is AI-assisted decision support only.
        """

        predicted_class = str(
            predicted_class
        ).lower()

        uncertainty_status = str(
            uncertainty_status
        ).upper()

        if uncertainty_status != "CONFIDENT":

            return (
                "The AI analysis shows uncertainty in the "
                "model output. The result should be reviewed "
                "by a qualified radiologist or treating "
                "physician before any medical decision is made. "
                "Clinical interpretation should consider the "
                "complete MRI study, medical history, and other "
                "relevant information."
            )

        if (
            tumor_detected
            and
            predicted_class == "no_tumor"
        ):

            return (
                "The segmentation model identified a "
                "tumor-like region, while the classification "
                "model assigned the highest probability to "
                "no_tumor. This difference between model "
                "outputs requires professional review. A "
                "qualified radiologist or treating physician "
                "should evaluate the complete MRI study and "
                "clinical context."
            )

        tumor_classes = {
            "glioma",
            "meningioma",
            "pituitary"
        }

        if predicted_class in tumor_classes:

            return (
                f"The AI classification model assigned the "
                f"highest probability to {predicted_class}. "
                "This is an AI-assisted model output and is "
                "not a confirmed diagnosis. Review by a "
                "qualified radiologist or treating physician "
                "is recommended for confirmation and clinical "
                "interpretation."
            )

        if (
            predicted_class == "no_tumor"
            and
            not tumor_detected
        ):

            return (
                "The AI classification model assigned the "
                "highest probability to no_tumor, and no "
                "tumor-like region was identified by the "
                "segmentation model. This result does not "
                "by itself rule out disease. Clinical "
                "interpretation by a qualified healthcare "
                "professional is recommended."
            )

        return (
            "The AI-assisted result should be reviewed by "
            "a qualified radiologist or treating physician. "
            "Clinical interpretation should consider the "
            "complete MRI study, medical history, and other "
            "relevant clinical information."
        )

    # ==========================================================
    # EXPLAINABILITY
    # ==========================================================

    def create_explainability_image(
        self,
        volume,
        predicted_mask,
        selected_slices,
        case_id
    ):
        """
        Create segmentation-based explainability image.

        This visualization is derived directly from the
        deployed U-Net segmentation output.

        It is NOT Grad-CAM.

        The image contains:
        - MRI slice
        - U-Net tumor mask
        - tumor localization overlay
        - tumor ROI bounding box
        """

        if not selected_slices:
            return None

        explainability_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            "reports",
            "explainability"
        )

        os.makedirs(
            explainability_dir,
            exist_ok=True
        )

        image_path = create_tumor_explanation(
            volume=volume,
            predicted_mask=predicted_mask,
            selected_slices=selected_slices,
            output_dir=explainability_dir,
            case_id=case_id
        )

        return image_path

    # ==========================================================
    # COMPLETE ANALYSIS
    # ==========================================================

    def analyze(
        self,
        volume: np.ndarray,
        case_id: str = "unknown",
        top_k: int = 7,
        min_pixels: int = 50,
        roi_margin: int = 20
    ):
        """
        Run the complete Edge AI inference pipeline.
        """

        if volume.shape != (
            4,
            240,
            240,
            155
        ):
            raise ValueError(
                "Expected MRI volume shape "
                "(4, 240, 240, 155), got "
                f"{volume.shape}"
            )

        volume = volume.astype(
            np.float32
        )

        print()
        print("=" * 70)
        print("EDGE AI PATIENT ANALYSIS")
        print("=" * 70)
        print("Case ID:", case_id)

        # ------------------------------------------------------
        # 1. Segmentation
        # ------------------------------------------------------

        predicted_mask = (
            self.segmenter.predict_volume(
                volume
            )
        )

        tumor_detected = bool(
            predicted_mask.sum() > 0
        )

        # ------------------------------------------------------
        # 2. Tumor slices
        # ------------------------------------------------------

        (
            tumor_slices,
            tumor_pixels_per_slice
        ) = self.get_tumor_slices(
            predicted_mask,
            min_pixels=min_pixels
        )

        if len(tumor_slices) == 0:

            recommendation = (
                "The segmentation model did not "
                "identify a sufficient tumor-like "
                "region for downstream classification. "
                "The complete MRI study should be "
                "reviewed by a qualified healthcare "
                "professional."
            )

            result = {
                "case_id": case_id,

                "images_shape":
                    tuple(volume.shape),

                "predicted_mask_shape":
                    tuple(predicted_mask.shape),

                "tumor_detected":
                    tumor_detected,

                "tumor_slice_count":
                    0,

                # Canonical frontend/report field
                "tumor_slices":
                    0,

                "selected_slices":
                    [],

                "slice_results":
                    [],

                "probabilities": {
                    class_name: 0.0
                    for class_name
                    in self.CLASS_NAMES
                },

                "prediction":
                    None,

                "confidence":
                    0.0,

                "entropy":
                    0.0,

                "normalized_entropy":
                    0.0,

                "margin":
                    0.0,

                # Canonical field
                "top2_margin":
                    0.0,

                "uncertainty_status":
                    "UNCERTAIN / REVIEW",

                "recommendation":
                    recommendation,

                # Canonical field
                "guidance":
                    recommendation,

                # No selected slice means no visualization.
                "explainability_image":
                    None,

                "predicted_mask":
                    predicted_mask
            }

            print()
            print("Tumor detected     :", tumor_detected)
            print("Tumor slices       :", 0)
            print("Selected slices    : []")
            print("Prediction         : None")
            print("Uncertainty status : UNCERTAIN / REVIEW")
            print()
            print("✓ Edge AI analysis completed")

            return result

        # ------------------------------------------------------
        # 3. Top-K slices
        # ------------------------------------------------------

        selected_slices = (
            self.select_top_tumor_slices(
                tumor_slices,
                tumor_pixels_per_slice,
                top_k=top_k
            )
        )

        # ------------------------------------------------------
        # 4. Classification
        # ------------------------------------------------------

        (
            slice_results,
            aggregated_probabilities,
            final_prediction,
            final_confidence
        ) = self.classify_selected_slices(
            volume,
            predicted_mask,
            selected_slices,
            roi_margin=roi_margin
        )

        # ------------------------------------------------------
        # 5. Uncertainty
        # ------------------------------------------------------

        (
            confidence,
            entropy,
            normalized_entropy,
            margin
        ) = self.calculate_uncertainty(
            aggregated_probabilities
        )

        uncertainty_status = (
            self.determine_uncertainty_status(
                confidence,
                normalized_entropy,
                margin
            )
        )

        # ------------------------------------------------------
        # 6. Guidance
        # ------------------------------------------------------

        recommendation = (
            self.generate_patient_guidance(
                tumor_detected=tumor_detected,
                predicted_class=final_prediction,
                uncertainty_status=uncertainty_status,
                confidence=confidence,
                top2_margin=margin
            )
        )

        # ------------------------------------------------------
        # 7. Explainability visualization
        # ------------------------------------------------------

        explainability_image = (
            self.create_explainability_image(
                volume=volume,
                predicted_mask=predicted_mask,
                selected_slices=selected_slices,
                case_id=case_id
            )
        )

        # ------------------------------------------------------
        # 8. Result
        # ------------------------------------------------------

        probability_dict = {
            class_name: float(probability)
            for class_name, probability
            in zip(
                self.CLASS_NAMES,
                aggregated_probabilities
            )
        }

        result = {
            "case_id":
                case_id,

            "images_shape":
                tuple(volume.shape),

            "predicted_mask_shape":
                tuple(predicted_mask.shape),

            "tumor_detected":
                tumor_detected,

            # Existing field
            "tumor_slice_count":
                int(len(tumor_slices)),

            # Canonical field for frontend/report
            "tumor_slices":
                int(len(tumor_slices)),

            "selected_slices":
                [int(x) for x in selected_slices],

            "slice_results":
                slice_results,

            "probabilities":
                probability_dict,

            "prediction":
                final_prediction,

            "confidence":
                float(confidence),

            "entropy":
                float(entropy),

            "normalized_entropy":
                float(normalized_entropy),

            # Existing field
            "margin":
                float(margin),

            # Canonical field
            "top2_margin":
                float(margin),

            "uncertainty_status":
                uncertainty_status,

            # Existing field
            "recommendation":
                recommendation,

            # Canonical field
            "guidance":
                recommendation,

            # New explainability output
            "explainability_image":
                explainability_image,

            # Internal field used by report generation
            "predicted_mask":
                predicted_mask
        }

        # ------------------------------------------------------
        # Console output
        # ------------------------------------------------------

        print()
        print(
            "Tumor detected     :",
            tumor_detected
        )

        print(
            "Tumor slices       :",
            len(tumor_slices)
        )

        print(
            "Selected slices    :",
            selected_slices
        )

        print(
            "Prediction         :",
            final_prediction
        )

        print(
            "Confidence         :",
            f"{confidence * 100:.2f}%"
        )

        print(
            "Normalized entropy :",
            f"{normalized_entropy:.4f}"
        )

        print(
            "Top-1/Top-2 margin :",
            f"{margin:.4f}"
        )

        print(
            "Uncertainty status :",
            uncertainty_status
        )

        print(
            "Explainability     :",
            explainability_image
        )

        print()
        print("✓ Edge AI analysis completed")

        return result