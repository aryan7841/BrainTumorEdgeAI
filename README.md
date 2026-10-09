\# BrainTumorEdgeAI

\### Edge AI-Based Brain Tumor Segmentation, Classification, Explainability, and Uncertainty-Aware Decision Support



BrainTumorEdgeAI is a research prototype that combines deep learning, ONNX-based inference, explainability visualizations, and uncertainty estimation to analyze brain MRI data.



The application provides a web interface for uploading MRI studies, viewing model-generated results, examining tumor localization, and downloading a PDF analysis report.



> \*\*Important:\*\* This project is an educational and research prototype. It is not a clinically validated diagnostic system and must not be used as a substitute for professional medical assessment.



\## Features



\- \*\*Tumor segmentation:\*\* Lightweight U-Net segments tumor-like regions in MRI slices.

\- \*\*AI classification:\*\* MobileNetV3-Small predicts one of four classes: glioma, meningioma, no tumor, or pituitary.

\- \*\*Multi-slice analysis:\*\* Selects relevant slices and aggregates classification probabilities.

\- \*\*Uncertainty estimation:\*\* Uses confidence, normalized entropy, and the top-two probability margin to flag results for review.

\- \*\*Explainability visualization:\*\* Displays MRI slices, predicted masks, localization overlays, and ROI bounding boxes.

\- \*\*PDF reporting:\*\* Generates a downloadable report with model outputs, uncertainty information, and review guidance.

\- \*\*Web application:\*\* React frontend with a FastAPI backend.

\- \*\*ONNX inference:\*\* Uses ONNX Runtime for CPU-based model inference.



\## System Architecture



1\. Upload a supported MRI study.

2\. Load the four MRI modalities: T1, T1ce, T2, and FLAIR.

3\. Apply the Lightweight U-Net segmentation model.

4\. Select tumor-containing slices and extract regions of interest.

5\. Convert the extracted regions into the input representation expected by the classifier.

6\. Run MobileNetV3-Small classification and aggregate probabilities across selected slices.

7\. Estimate uncertainty and generate an explainability visualization.

8\. Display the results and provide a PDF report.



\## Technology Stack



| Component | Technology |

|---|---|

| Frontend | React, Vite, JavaScript |

| Backend API | Python, FastAPI |

| Segmentation | Lightweight U-Net, ONNX Runtime |

| Classification | MobileNetV3-Small, ONNX Runtime |

| Image processing | NumPy, OpenCV |

| PDF reports | ReportLab |

| Visualization | Matplotlib |

| Datasets | BraTS2021, BRISC2025 |



\## Research Results



\### Segmentation — BraTS2021



\- Best validation Dice score: \*\*0.8801\*\*

\- Held-out test Dice: \*\*0.8903\*\*

\- Held-out test IoU: \*\*0.8611\*\*

\- Held-out test precision: \*\*0.9723\*\*

\- Held-out test recall: \*\*0.8826\*\*



\### Classification — BRISC2025



\- Standalone test accuracy: \*\*97.90%\*\*

\- Macro F1-score: \*\*97.96%\*\*

\- Weighted F1-score: \*\*97.89%\*\*



These results come from separate segmentation and classification experiments. They do not establish equivalent performance for the complete integrated pipeline on clinical data.



\## Project Structure



```text

BrainTumorEdgeAI/

├── backend/

│   ├── api/

│   ├── explainability/

│   ├── inference/

│   ├── models/

│   ├── reporting/

│   ├── test\_classification.py

│   ├── test\_pipeline.py

│   ├── test\_report.py

│   └── test\_segmentation.py

├── frontend/

│   ├── public/

│   ├── src/

│   ├── package.json

│   └── package-lock.json

├── data/                 # Local datasets; excluded from Git

├── notebooks/

├── reports/               # Generated reports; excluded from Git

├── scripts/

├── tests/

├── requirements.txt

├── .gitignore

└── README.md

```



\## Prerequisites



\- Python 3.12 recommended

\- Node.js and npm

\- Git

\- The model files required by the backend



\## Installation and Setup



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd BrainTumorEdgeAI

```



Replace `YOUR\_GITHUB\_REPOSITORY\_URL` with the repository URL after publishing.



\### 2. Create and activate a Python environment



Windows PowerShell:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install Python dependencies



```powershell

python -m pip install -r requirements.txt

```



\### 4. Install frontend dependencies



```powershell

cd frontend

npm install

cd ..

```



\### 5. Start the backend



From the project root:



```powershell

python -m uvicorn backend.api.main:app --reload

```



The backend should be available at:



\- API: http://127.0.0.1:8000

\- API documentation: http://127.0.0.1:8000/docs



\### 6. Start the frontend



Open a second terminal:



```powershell

cd frontend

npm run dev

```



Open the local URL printed by Vite in your terminal.



\## Model Files



The application uses ONNX model files for inference. The repository is intended to include:



\- `backend/models/unet\_brain\_tumor\_int8.onnx`

\- `backend/models/mobilenetv3\_small\_brain\_tumor.onnx`



Ensure these files are present in the cloned project. Before distributing model files publicly, check the relevant dataset terms, model provenance, and redistribution permissions.



\## Limitations



\- The segmentation model was trained using BraTS2021, while the classification model was trained using BRISC2025.

\- The integrated pipeline uses a cross-dataset preprocessing and representation conversion strategy that has not been clinically validated.

\- Classification predictions on BraTS cases are exploratory model outputs; the integrated evaluation does not provide ground-truth tumor-type accuracy.

\- Uncertainty thresholds are prototype engineering settings, not clinically validated thresholds.

\- Explainability visualizations support interpretation of model outputs but do not establish diagnostic correctness.

\- Performance on the development machine does not guarantee equivalent performance on other edge devices.

\- This prototype is not approved for clinical diagnosis or treatment decisions.



\## Responsible Use



This project is intended for educational research, software development, and experimentation. MRI analysis and generated reports should be reviewed by qualified medical professionals. Do not use the outputs alone to make medical decisions.



\## Future Work



\- Validate the complete pipeline on appropriately labeled, independent datasets.

\- Evaluate cross-dataset generalization and improve modality compatibility.

\- Validate uncertainty thresholds using suitable evaluation protocols.

\- Test deployment on representative edge hardware.

\- Improve usability, security, and handling of sensitive medical data.



\## License



A license has not yet been selected. Add a suitable license only after confirming that you have the rights to publish the code, model files, and any other included material.

