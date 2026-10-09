import os
import cv2
import numpy as np
import onnxruntime as ort


class BrainTumorClassifier:
    """
    Edge AI brain tumor classification using
    MobileNetV3-Small ONNX.
    """

    CLASSES = [
        "glioma",
        "meningioma",
        "no_tumor",
        "pituitary"
    ]

    IMAGE_SIZE = 224

    # ImageNet normalization used by the trained classifier
    MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    def __init__(self, model_path: str):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Classification model not found: {model_path}"
            )

        self.model_path = model_path

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Classification model loaded successfully.")
        print(f"Model: {model_path}")
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        Convert a grayscale ROI into the 3-channel format
        expected by MobileNetV3-Small.

        Input:
            roi -> 2D grayscale image

        Output:
            (1, 3, 224, 224) float32 tensor
        """

        if roi is None:
            raise ValueError("ROI is None.")

        if roi.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale ROI, got {roi.shape}"
            )

        roi = roi.astype(np.float32)

        # Min-max normalization
        roi_min = roi.min()
        roi_max = roi.max()

        if roi_max > roi_min:
            roi = (
                roi - roi_min
            ) / (
                roi_max - roi_min
            )
        else:
            roi = np.zeros_like(roi)

        # Convert [0,1] → [0,255]
        roi_uint8 = (
            roi * 255.0
        ).clip(0, 255).astype(np.uint8)

        # Resize
        roi_resized = cv2.resize(
            roi_uint8,
            (
                self.IMAGE_SIZE,
                self.IMAGE_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # Grayscale → 3 channels
        image = np.stack(
            [roi_resized] * 3,
            axis=-1
        )

        # Convert to float [0,1]
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        image = (
            image - self.MEAN
        ) / self.STD

        # HWC → CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        return image.astype(np.float32)

    def predict(self, roi: np.ndarray) -> dict:
        """
        Classify one ROI.

        Returns:
            prediction
            confidence
            probabilities
        """

        input_tensor = self.preprocess(roi)

        logits = self.session.run(
            [self.output_name],
            {
                self.input_name: input_tensor
            }
        )[0]

        logits = logits[0]

        # Stable softmax
        exp_logits = np.exp(
            logits - np.max(logits)
        )

        probabilities = (
            exp_logits /
            np.sum(exp_logits)
        )

        predicted_index = int(
            np.argmax(probabilities)
        )

        prediction = self.CLASSES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        probability_dict = {
            class_name: float(prob)
            for class_name, prob in zip(
                self.CLASSES,
                probabilities
            )
        }

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probability_dict
        }