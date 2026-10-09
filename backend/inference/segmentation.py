import os
import numpy as np
import onnxruntime as ort


class BrainTumorSegmenter:
    """
    Edge AI brain tumor segmentation using
    the INT8 ONNX Lightweight U-Net.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Segmentation model not found: {model_path}"
            )

        self.model_path = model_path

        # Use CPU execution for local Edge AI deployment
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Segmentation model loaded successfully.")
        print(f"Model: {model_path}")
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_name}")

    def predict_slice(self, image_slice: np.ndarray) -> np.ndarray:
        """
        Run segmentation on one MRI slice.

        Expected input:
            image_slice -> (4, 240, 240)

        Returns:
            binary tumor mask -> (240, 240)
        """

        if image_slice.shape != (4, 240, 240):
            raise ValueError(
                f"Expected slice shape (4, 240, 240), "
                f"got {image_slice.shape}"
            )

        # Convert to float32
        image_slice = image_slice.astype(np.float32)

        # Add batch dimension
        input_tensor = np.expand_dims(image_slice, axis=0)

        # ONNX inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        # Remove batch/channel dimensions
        logits = output[0, 0]

        # Convert logits to probability
        probability = 1.0 / (1.0 + np.exp(-logits))

        # Binary segmentation mask
        mask = (probability >= 0.5).astype(np.uint8)

        return mask

    def predict_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a complete 3D MRI volume.

        Expected input:
            volume -> (4, 240, 240, 155)

        Returns:
            tumor mask -> (240, 240, 155)
        """

        if volume.shape != (4, 240, 240, 155):
            raise ValueError(
                f"Expected volume shape (4, 240, 240, 155), "
                f"got {volume.shape}"
            )

        masks = []

        print("Starting Edge AI segmentation...")

        for slice_index in range(volume.shape[-1]):

            image_slice = volume[:, :, :, slice_index]

            mask = self.predict_slice(image_slice)

            masks.append(mask)

            if (slice_index + 1) % 10 == 0:
                print(
                    f"Processed {slice_index + 1}/"
                    f"{volume.shape[-1]} slices"
                )

        # Stack slices along Z-axis
        mask_3d = np.stack(masks, axis=-1)

        print("Segmentation completed.")
        print(f"Output mask shape: {mask_3d.shape}")

        return mask_3d