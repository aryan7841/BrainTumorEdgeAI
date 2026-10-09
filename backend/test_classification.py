import os
import numpy as np

from inference.classification import BrainTumorClassifier


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "mobilenetv3_small_brain_tumor.onnx"
)


print("=" * 60)
print("EDGE AI - MOBILENET CLASSIFICATION TEST")
print("=" * 60)


classifier = BrainTumorClassifier(
    MODEL_PATH
)


# Fake grayscale ROI for deployment pipeline test
roi = np.random.rand(
    180,
    180
).astype(np.float32)


print("\nROI shape:", roi.shape)


result = classifier.predict(roi)


print("\nPrediction:")
print(result["prediction"])

print("\nConfidence:")
print(
    f"{result['confidence'] * 100:.2f}%"
)

print("\nClass probabilities:")

for class_name, probability in result[
    "probabilities"
].items():

    print(
        f"{class_name:12s}: "
        f"{probability * 100:.2f}%"
    )


# Validation

assert result["prediction"] in classifier.CLASSES

assert 0.0 <= result["confidence"] <= 1.0

assert len(
    result["probabilities"]
) == 4

assert abs(
    sum(result["probabilities"].values()) - 1.0
) < 1e-5


print("\n" + "=" * 60)
print("MOBILENET CLASSIFICATION TEST PASSED")
print("=" * 60)