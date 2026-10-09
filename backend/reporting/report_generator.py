import os

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


class BrainTumorReportGenerator:
    """
    BrainTumorEdgeAI PDF report generator.

    Report structure follows the project's Notebook 6
    clinical decision-support report:

    1. Case Information
    2. MRI Analysis Summary
    3. AI Classification Output
    4. Uncertainty Analysis
    5. Model Output Consistency
    6. Explainable AI
    7. AI-Assisted Finding
    8. Suggested Next Steps
    9. Important Disclaimer

    ReportLab is used for Windows-compatible PDF generation.
    """

    def __init__(self, output_dir: str):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=21,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=6
        )

        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=5
        )

        self.heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8
        )

        self.body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=7
        )

        self.small_style = ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.grey
        )

        self.warning_style = ParagraphStyle(
            "Warning",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=7
        )

        self.prediction_style = ParagraphStyle(
            "Prediction",
            parent=styles["BodyText"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=10
        )

    # =====================================================
    # PUBLIC METHOD
    # =====================================================

    def generate(
        self,
        report_data: dict,
        pipeline_image_path=None,
        uncertainty_image_path=None,
        gradcam_image_path=None
    ) -> dict:

        data = report_data.copy()

        # -------------------------------------------------
        # Defaults
        # -------------------------------------------------

        case_id = str(
            data.get(
                "case_id",
                "Unknown"
            )
        )

        data.setdefault(
            "report_title",
            "AI-Assisted Brain MRI Analysis Report"
        )

        data.setdefault(
            "patient_id",
            case_id
        )

        data.setdefault(
            "study_id",
            case_id
        )

        data.setdefault(
            "modalities",
            [
                "T1",
                "T1CE",
                "T2",
                "FLAIR"
            ]
        )

        data.setdefault(
            "tumor_detected",
            False
        )

        data.setdefault(
            "selected_slices",
            []
        )

        data.setdefault(
            "probabilities",
            {}
        )

        # Current pipeline field name
        data.setdefault(
            "prediction",
            "unknown"
        )

        # Current pipeline field name
        data.setdefault(
            "tumor_slice_count",
            0
        )

        # Current pipeline field name
        data.setdefault(
            "margin",
            0.0
        )

        data.setdefault(
            "confidence",
            0.0
        )

        data.setdefault(
            "normalized_entropy",
            0.0
        )

        data.setdefault(
            "uncertainty_status",
            "UNKNOWN"
        )

        data.setdefault(
            "recommendation",
            (
                "The AI-assisted result should be reviewed "
                "by a qualified radiologist or treating "
                "physician. Clinical interpretation should "
                "consider the complete MRI study, medical "
                "history, and other relevant clinical "
                "information."
            )
        )

        data.setdefault(
            "disclaimer",
            (
                "This report is generated by an AI research "
                "prototype and is intended for research and "
                "decision-support demonstration only. It is "
                "not a medical diagnosis and must not be used "
                "as a substitute for evaluation by a qualified "
                "healthcare professional."
            )
        )

        # -------------------------------------------------
        # IMPORTANT FIELD-NAME COMPATIBILITY
        # -------------------------------------------------

        # Current pipeline:
        # tumor_slice_count
        #
        # Older report generator:
        # tumor_slices
        #
        # Keep both available.
        data["tumor_slices"] = data.get(
            "tumor_slice_count",
            data.get("tumor_slices", 0)
        )

        # Current pipeline:
        # prediction
        #
        # Older report generator:
        # predicted_class
        #
        # Keep both available.
        data["predicted_class"] = data.get(
            "prediction",
            data.get("predicted_class", "unknown")
        )

        # Current pipeline:
        # margin
        #
        # Older report generator:
        # top2_margin
        #
        # Keep both available.
        data["top2_margin"] = data.get(
            "margin",
            data.get("top2_margin", 0.0)
        )

        # -------------------------------------------------
        # Derived values
        # -------------------------------------------------

        tumor_detected = bool(
            data["tumor_detected"]
        )

        confidence = float(
            data["confidence"]
        )

        normalized_entropy = float(
            data["normalized_entropy"]
        )

        top2_margin = float(
            data["top2_margin"]
        )

        probabilities = data["probabilities"]

        confidence_percent = (
            confidence * 100
        )

        modalities = data["modalities"]

        if isinstance(
            modalities,
            (list, tuple)
        ):

            modalities_text = ", ".join(
                str(x)
                for x in modalities
            )

        else:

            modalities_text = str(
                modalities
            )

        selected_slices = data[
            "selected_slices"
        ]

        if isinstance(
            selected_slices,
            (list, tuple)
        ):

            selected_slices_text = ", ".join(
                str(x)
                for x in selected_slices
            )

        else:

            selected_slices_text = str(
                selected_slices
            )

        # -------------------------------------------------
        # Safe filename
        # -------------------------------------------------

        safe_case_id = "".join(
            character
            for character in case_id
            if character.isalnum()
            or character in "_-"
        )

        if not safe_case_id:
            safe_case_id = "unknown_case"

        html_path = os.path.join(
            self.output_dir,
            f"{safe_case_id}_AI_Report.html"
        )

        pdf_path = os.path.join(
            self.output_dir,
            f"{safe_case_id}_AI_Report.pdf"
        )

        # =================================================
        # PDF DOCUMENT
        # =================================================

        document = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        story = []

        # =================================================
        # HEADER
        # =================================================

        story.append(
            Paragraph(
                data["report_title"],
                self.title_style
            )
        )

        story.append(
            Paragraph(
                "AI-Assisted Brain MRI Analysis",
                self.subtitle_style
            )
        )

        story.append(
            Paragraph(
                "Research / Decision-Support Prototype",
                self.subtitle_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

        # =================================================
        # 1. CASE INFORMATION
        # =================================================

        story.append(
            Paragraph(
                "1. Case Information",
                self.heading_style
            )
        )

        case_rows = [
            [
                "Patient / Case ID",
                str(data["patient_id"])
            ],
            [
                "Study ID",
                str(data["study_id"])
            ],
            [
                "Report Date",
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                )
            ],
            [
                "MRI Modalities",
                modalities_text
            ]
        ]

        story.append(
            self._make_info_table(
                case_rows
            )
        )

        # =================================================
        # 2. MRI ANALYSIS SUMMARY
        # =================================================

        story.append(
            Paragraph(
                "2. MRI Analysis Summary",
                self.heading_style
            )
        )

        segmentation_status = (
            "Yes"
            if tumor_detected
            else "No"
        )

        segmentation_rows = [
            [
                "Segmentation Model",
                "Lightweight U-Net"
            ],
            [
                "AI-Detected Tumor Region",
                segmentation_status
            ],
            [
                "Tumor-Containing Slices",
                str(
                    data.get(
                        "tumor_slices",
                        0
                    )
                )
            ],
            [
                "Selected Slices",
                selected_slices_text
            ]
        ]

        story.append(
            self._make_info_table(
                segmentation_rows
            )
        )

        # -------------------------------------------------
        # Explainability / localization image
        # -------------------------------------------------

        if pipeline_image_path:

            image = self._make_image(
                pipeline_image_path,
                max_width=6.2 * inch,
                max_height=3.2 * inch
            )

            if image:

                story.append(
                    Spacer(1, 8)
                )

                story.append(
                    image
                )

        # =================================================
        # 3. AI CLASSIFICATION OUTPUT
        # =================================================

        story.append(
            Paragraph(
                "3. AI Classification Output",
                self.heading_style
            )
        )

        story.append(
            Paragraph(
                "<b>Highest-Probability Class</b>",
                self.body_style
            )
        )

        story.append(
            Paragraph(
                str(
                    data["predicted_class"]
                ),
                self.prediction_style
            )
        )

        classification_rows = [
            [
                "Top-1 Confidence",
                f"{confidence_percent:.2f}%"
            ]
        ]

        story.append(
            self._make_info_table(
                classification_rows
            )
        )

        story.append(
            Spacer(1, 8)
        )

        probability_rows = [
            [
                "Class",
                "Probability"
            ]
        ]

        for class_name, probability in probabilities.items():

            probability_rows.append(
                [
                    str(class_name),
                    f"{float(probability) * 100:.2f}%"
                ]
            )

        probability_table = Table(
            probability_rows,
            colWidths=[
                3.2 * inch,
                3.0 * inch
            ]
        )

        probability_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ]
            )
        )

        story.append(
            probability_table
        )

        # =================================================
        # 4. UNCERTAINTY ANALYSIS
        # =================================================

        story.append(
            Paragraph(
                "4. Uncertainty Analysis",
                self.heading_style
            )
        )

        uncertainty_rows = [
            [
                "Confidence",
                f"{confidence_percent:.2f}%"
            ],
            [
                "Normalized Entropy",
                f"{normalized_entropy:.4f}"
            ],
            [
                "Top-1 / Top-2 Margin",
                f"{top2_margin:.4f}"
            ],
            [
                "Prototype Uncertainty Status",
                str(
                    data["uncertainty_status"]
                )
            ]
        ]

        story.append(
            self._make_info_table(
                uncertainty_rows
            )
        )

        if uncertainty_image_path:

            image = self._make_image(
                uncertainty_image_path,
                max_width=6.2 * inch,
                max_height=3.2 * inch
            )

            if image:

                story.append(
                    Spacer(1, 8)
                )

                story.append(
                    image
                )

        warning_box = Table(
            [[
                Paragraph(
                    (
                        "The uncertainty thresholds used by this "
                        "system are prototype engineering thresholds "
                        "and have not been clinically validated."
                    ),
                    self.body_style
                )
            ]],
            colWidths=[
                6.2 * inch
            ]
        )

        warning_box.setStyle(
            TableStyle(
                [
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        colors.grey
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        colors.whitesmoke
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    )
                ]
            )
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            warning_box
        )

        # =================================================
        # 5. MODEL OUTPUT CONSISTENCY
        # =================================================

        story.append(
            Paragraph(
                "5. Model Output Consistency",
                self.heading_style
            )
        )

        predicted_class = str(
            data["predicted_class"]
        )

        if (
            tumor_detected
            and predicted_class == "no_tumor"
        ):

            consistency_text = (
                "<b>Review Flag:</b><br/><br/>"
                "The segmentation model identified a "
                "tumor-like region, while the classification "
                "model produced <b>no_tumor</b> as its "
                "highest-probability class.<br/><br/>"
                "This difference between model outputs requires "
                "review by a qualified healthcare professional "
                "and should not be interpreted as a definitive "
                "diagnosis."
            )

        elif (
            not tumor_detected
            and predicted_class != "no_tumor"
        ):

            consistency_text = (
                "<b>Review Flag:</b><br/><br/>"
                "The segmentation model did not identify a "
                "tumor-like region while the classification "
                "model assigned the highest probability to a "
                "tumor class.<br/><br/>"
                "Professional review is recommended."
            )

        else:

            consistency_text = (
                "The segmentation and classification outputs "
                "do not show a direct output conflict in this "
                "case."
            )

        consistency_box = Table(
            [[
                Paragraph(
                    consistency_text,
                    self.body_style
                )
            ]],
            colWidths=[
                6.2 * inch
            ]
        )

        consistency_box.setStyle(
            TableStyle(
                [
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    )
                ]
            )
        )

        story.append(
            consistency_box
        )

        # =================================================
        # 6. EXPLAINABLE AI
        # =================================================

        story.append(
            Paragraph(
                "6. Explainable AI",
                self.heading_style
            )
        )

        # The current deployment uses segmentation-based
        # tumor localization explainability.
        if pipeline_image_path:

            explainability_text = (
                "<b>U-Net tumor localization visualization "
                "available.</b><br/><br/>"
                "The visualization shows the MRI slice, "
                "predicted tumor mask, tumor localization "
                "overlay, and detected tumor ROI bounding box."
                "<br/><br/>"
                "This visualization is an interpretability aid "
                "and is not clinical ground truth."
            )

        else:

            explainability_text = (
                "U-Net tumor localization visualization "
                "was not available for this case."
            )

        story.append(
            Paragraph(
                explainability_text,
                self.body_style
            )
        )

        # Optional Grad-CAM
        if gradcam_image_path:

            image = self._make_image(
                gradcam_image_path,
                max_width=6.2 * inch,
                max_height=3.2 * inch
            )

            if image:

                story.append(
                    Spacer(1, 8)
                )

                story.append(
                    image
                )

                story.append(
                    Paragraph(
                        (
                            "Grad-CAM provides a visualization "
                            "of image regions that contributed "
                            "to the classifier output. It is "
                            "not a tumor segmentation mask or "
                            "clinical ground truth."
                        ),
                        self.small_style
                    )
                )

        # =================================================
        # 7. AI-ASSISTED FINDING
        # =================================================

        story.append(
            Paragraph(
                "7. AI-Assisted Finding",
                self.heading_style
            )
        )

        finding_text = (
            "The AI classification model assigned the "
            "highest probability to:<br/><br/>"
            f"<b>{predicted_class}</b><br/><br/>"
            "This represents the output of the trained AI "
            "model and is not, by itself, a medical diagnosis."
        )

        story.append(
            Paragraph(
                finding_text,
                self.body_style
            )
        )

        # =================================================
        # 8. SUGGESTED NEXT STEPS
        # =================================================

        story.append(
            Paragraph(
                "8. Suggested Next Steps",
                self.heading_style
            )
        )

        recommendation_box = Table(
            [[
                Paragraph(
                    str(
                        data["recommendation"]
                    ),
                    self.body_style
                )
            ]],
            colWidths=[
                6.2 * inch
            ]
        )

        recommendation_box.setStyle(
            TableStyle(
                [
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    )
                ]
            )
        )

        story.append(
            recommendation_box
        )

        # =================================================
        # 9. IMPORTANT DISCLAIMER
        # =================================================

        story.append(
            Paragraph(
                "9. Important Disclaimer",
                self.heading_style
            )
        )

        disclaimer_box = Table(
            [[
                Paragraph(
                    str(
                        data["disclaimer"]
                    ),
                    self.small_style
                )
            ]],
            colWidths=[
                6.2 * inch
            ]
        )

        disclaimer_box.setStyle(
            TableStyle(
                [
                    (
                        "LINEABOVE",
                        (0, 0),
                        (-1, 0),
                        0.8,
                        colors.grey
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        10
                    )
                ]
            )
        )

        story.append(
            disclaimer_box
        )

        # =================================================
        # FOOTER
        # =================================================

        story.append(
            Spacer(1, 15)
        )

        story.append(
            Paragraph(
                "Generated by BrainTumorEdgeAI | "
                "AI Research Prototype | "
                "ONNX Runtime CPU",
                self.small_style
            )
        )

        # =================================================
        # BUILD PDF
        # =================================================

        document.build(
            story
        )

        # =================================================
        # SAVE SIMPLE HTML COPY
        # =================================================

        self._save_html_summary(
            data,
            html_path
        )

        print("=" * 70)
        print("PATIENT PDF REPORT GENERATED")
        print("=" * 70)

        print(
            f"Case ID : {case_id}"
        )

        print(
            f"HTML    : {html_path}"
        )

        print(
            f"PDF     : {pdf_path}"
        )

        print(
            f"Size    : "
            f"{os.path.getsize(pdf_path) / 1024:.2f} KB"
        )

        return {
            "pdf_path": pdf_path,
            "html_path": html_path
        }

    # =====================================================
    # INFO TABLE
    # =====================================================

    @staticmethod
    def _make_info_table(rows):

        table = Table(
            rows,
            colWidths=[
                2.3 * inch,
                3.9 * inch
            ]
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        7
                    )
                ]
            )
        )

        return table

    # =====================================================
    # IMAGE HELPER
    # =====================================================

    @staticmethod
    def _make_image(
        image_path,
        max_width,
        max_height
    ):

        if not image_path:
            return None

        if not os.path.exists(image_path):
            return None

        try:

            image = Image(
                image_path
            )

            width = float(
                image.imageWidth
            )

            height = float(
                image.imageHeight
            )

            scale = min(
                max_width / width,
                max_height / height,
                1.0
            )

            image.drawWidth = (
                width * scale
            )

            image.drawHeight = (
                height * scale
            )

            return image

        except Exception as e:

            print(
                f"Warning: could not add "
                f"image {image_path}: {e}"
            )

            return None

    # =====================================================
    # SIMPLE HTML COPY
    # =====================================================

    @staticmethod
    def _save_html_summary(
        data,
        html_path
    ):

        probabilities = data.get(
            "probabilities",
            {}
        )

        rows = ""

        for name, probability in probabilities.items():

            rows += (
                "<tr>"
                f"<td>{name}</td>"
                f"<td>{float(probability) * 100:.2f}%</td>"
                "</tr>"
            )

        predicted_class = data.get(
            "prediction",
            data.get(
                "predicted_class",
                "unknown"
            )
        )

        tumor_slice_count = data.get(
            "tumor_slice_count",
            data.get(
                "tumor_slices",
                0
            )
        )

        margin = data.get(
            "margin",
            data.get(
                "top2_margin",
                0.0
            )
        )

        html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
{data.get(
    "report_title",
    "AI-Assisted Brain MRI Analysis Report"
)}
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #222;
}}

h1 {{
    text-align: center;
}}

h2 {{
    border-bottom: 1px solid #999;
    padding-bottom: 5px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}}

td, th {{
    border: 1px solid #ccc;
    padding: 8px;
}}

th {{
    background: #eee;
}}

.warning {{
    border: 1px solid #999;
    padding: 12px;
}}

</style>

</head>

<body>

<h1>
{data.get(
    "report_title",
    "AI-Assisted Brain MRI Analysis Report"
)}
</h1>

<p>
<b>Patient / Case ID:</b>
{data.get("patient_id", data.get("case_id", "Unknown"))}
</p>

<p>
<b>Study ID:</b>
{data.get("study_id", data.get("case_id", "Unknown"))}
</p>

<h2>
1. Case Information
</h2>

<p>
<b>MRI Modalities:</b>
T1, T1CE, T2, FLAIR
</p>

<h2>
2. MRI Analysis Summary
</h2>

<p>
<b>Tumor-like region detected:</b>
{"Yes" if data.get("tumor_detected") else "No"}
</p>

<p>
<b>Tumor-containing slices:</b>
{tumor_slice_count}
</p>

<p>
<b>Selected slices:</b>
{", ".join(str(x) for x in data.get("selected_slices", []))}
</p>

<h2>
3. AI Classification Output
</h2>

<p>
<b>Prediction:</b>
{predicted_class}
</p>

<p>
<b>Confidence:</b>
{float(data.get("confidence", 0)) * 100:.2f}%
</p>

<table>

<tr>
<th>Class</th>
<th>Probability</th>
</tr>

{rows}

</table>

<h2>
4. Uncertainty Analysis
</h2>

<p>
<b>Status:</b>
{data.get("uncertainty_status", "UNKNOWN")}
</p>

<p>
<b>Normalized Entropy:</b>
{data.get("normalized_entropy", 0)}
</p>

<p>
<b>Top-1 / Top-2 Margin:</b>
{margin}
</p>

<h2>
5. Model Output Consistency
</h2>

<div class="warning">

Segmentation and classification outputs should be
reviewed together. Any disagreement requires
professional review.

</div>

<h2>
6. Explainable AI
</h2>

<p>
U-Net tumor localization visualization is an
interpretability aid and is not clinical ground truth.
Grad-CAM is an optional classifier interpretability
method and is not a tumor segmentation mask.
</p>

<h2>
7. AI-Assisted Finding
</h2>

<p>
The highest-probability classification output is:
<b>{predicted_class}</b>
</p>

<h2>
8. Suggested Next Steps
</h2>

<p>
{data.get("recommendation", "")}
</p>

<h2>
9. Important Disclaimer
</h2>

<p>
{data.get("disclaimer", "")}
</p>

</body>

</html>
"""

        with open(
            html_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                html
            )