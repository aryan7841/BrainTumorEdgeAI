print("TEST SCRIPT STARTED")

import os
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"D:\Major Project\backend\models\unet_brain_tumor_int8.onnx"

print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("INT8 U-Net model not found.")

print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("Model loaded successfully!")

print("Providers:", session.get_providers())

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("Input name :", input_info.name)
print("Input shape:", input_info.shape)

print("Output name :", output_info.name)
print("Output shape:", output_info.shape)

# Create one fake MRI slice
test_input = np.random.randn(
    1, 4, 240, 240
).astype(np.float32)

print("Test input shape:", test_input.shape)

# Run inference
output = session.run(
    [output_info.name],
    {input_info.name: test_input}
)[0]

print("Output shape:", output.shape)

# Convert logits to probability
logits = output[0, 0]

probability = 1 / (1 + np.exp(-logits))

mask = (probability >= 0.5).astype(np.uint8)

print("Mask shape:", mask.shape)
print("Tumor pixels:", int(mask.sum()))
print("Unique mask values:", np.unique(mask))

print()
print("===================================")
print("INT8 U-NET TEST PASSED")
print("===================================")