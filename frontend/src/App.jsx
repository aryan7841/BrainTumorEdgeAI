import { useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    setFile(selectedFile || null);
    setResult(null);
    setError("");
  };

  const analyzeMRI = async () => {
    if (!file) {
      setError("Please select an MRI .npz file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_BASE_URL}/analyze`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "MRI analysis failed."
        );
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message || "Unable to connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    /*
      Backend currently returns the generated PDF path.

      Example:
      D:\Major Project\backend\reports\BraTS2021_00019_AI_Report.pdf
    */

    const pdfPath =
      result?.report?.pdf ||
      result?.report?.pdf_path;

    if (!pdfPath) {
      setError("PDF report path was not returned by the backend.");
      return;
    }

    setDownloading(true);
    setError("");

    try {
      /*
        Extract only the filename from a Windows path.

        Example:
        D:\...\BraTS2021_00019_AI_Report.pdf
        ->
        BraTS2021_00019_AI_Report.pdf
      */

      const filename = pdfPath
        .split("\\")
        .pop()
        .split("/")
        .pop();

      if (!filename) {
        throw new Error("Invalid PDF filename.");
      }

      /*
        Ask FastAPI for the actual PDF file.
      */

      const response = await fetch(
        `${API_BASE_URL}/reports/${encodeURIComponent(filename)}`
      );

      if (!response.ok) {
        throw new Error(
          `PDF download failed (${response.status}).`
        );
      }

      /*
        Convert the server response into a browser Blob.
      */

      const blob = await response.blob();

      /*
        Create a temporary browser URL.
      */

      const blobUrl = window.URL.createObjectURL(blob);

      /*
        Create an invisible download link.
      */

      const link = document.createElement("a");

      link.href = blobUrl;
      link.download = filename;

      document.body.appendChild(link);

      link.click();

      document.body.removeChild(link);

      /*
        Release temporary browser memory.
      */

      window.URL.revokeObjectURL(blobUrl);

    } catch (err) {

      console.error("PDF download error:", err);

      setError(
        err.message ||
        "Unable to download the PDF report."
      );

    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="app">

      {/* =================================================
          HEADER
      ================================================= */}

      <header className="header">

        <div>
          <h1>BrainTumorEdgeAI</h1>

          <p>
            Edge AI-Based Brain MRI Analysis
          </p>
        </div>

        <div className="status">

          <span className="status-dot"></span>

          Edge AI System

        </div>

      </header>


      {/* =================================================
          MAIN
      ================================================= */}

      <main className="main">

        {/* =================================================
            HERO
        ================================================= */}

        <section className="hero">

          <h2>
            AI-Assisted Brain MRI Analysis
          </h2>

          <p>
            Upload an MRI study for AI-assisted
            segmentation, classification,
            uncertainty analysis and clinical
            decision support.
          </p>

        </section>


        {/* =================================================
            UPLOAD CARD
        ================================================= */}

        <section className="card upload-card">

          <h3>
            Upload MRI Study
          </h3>

          <p className="description">

            Currently supported format:
            <strong> .npz</strong>

          </p>

          <div className="upload-area">

            <input
              type="file"
              accept=".npz"
              onChange={handleFileChange}
            />

            {file && (

              <div className="selected-file">

                Selected:
                {" "}
                <strong>
                  {file.name}
                </strong>

              </div>

            )}

          </div>

          <button
            className="analyze-button"
            onClick={analyzeMRI}
            disabled={loading || !file}
          >

            {loading
              ? "Analyzing MRI..."
              : "Analyze MRI"}

          </button>

        </section>


        {/* =================================================
            ERROR
        ================================================= */}

        {error && (

          <section className="error-card">

            <strong>
              Error:
            </strong>

            {" "}

            {error}

          </section>

        )}


        {/* =================================================
            RESULTS
        ================================================= */}

        {result && (

          <section className="results">

            <h2>
              Analysis Results
            </h2>


            {/* =================================================
                SUMMARY CARDS
            ================================================= */}

            <div className="result-grid">

              <div className="result-card">

                <span className="label">
                  Tumor Detection
                </span>

                <strong className="value">

                  {result.tumor_detected
                    ? "Tumor-like Region Detected"
                    : "No Tumor Detected"}

                </strong>

              </div>


              <div className="result-card">

                <span className="label">
                  AI Classification
                </span>

                <strong className="value">

                  {result.prediction || "Unknown"}

                </strong>

              </div>


              <div className="result-card">

                <span className="label">
                  Confidence
                </span>

                <strong className="value">

                  {(Number(result.confidence || 0) * 100).toFixed(2)}%

                </strong>

              </div>


              <div className="result-card">

                <span className="label">
                  Uncertainty Status
                </span>

                <strong className="value">

                  {result.uncertainty_status || "UNKNOWN"}

                </strong>

              </div>

            </div>


            {/* =================================================
                MRI ANALYSIS SUMMARY
            ================================================= */}

            <div className="card">

              <h3>
                MRI Analysis Summary
              </h3>

              <div className="details">

                <div>

                  <span>
                    Tumor slices
                  </span>

                  <strong>

                    {result.tumor_slices ??
                      result.tumor_slice_count ??
                      0}

                  </strong>

                </div>


                <div>

                  <span>
                    Selected slices
                  </span>

                  <strong>

                    {result.selected_slices?.join(", ") ||
                      "None"}

                  </strong>

                </div>


                <div>

                  <span>
                    Normalized entropy
                  </span>

                  <strong>

                    {Number(
                      result.normalized_entropy || 0
                    ).toFixed(4)}

                  </strong>

                </div>


                <div>

                  <span>
                    Top-1 / Top-2 margin
                  </span>

                  <strong>

                    {Number(
                      result.top2_margin ??
                      result.margin ??
                      0
                    ).toFixed(4)}

                  </strong>

                </div>

              </div>

            </div>


            {/* =================================================
                EXPLAINABILITY
            ================================================= */}

            {result.explainability?.available && (

              <div className="card">

                <h3>
                  Explainable AI
                </h3>

                <p className="description">

                  U-Net tumor localization visualization
                  generated from the segmentation output.

                </p>

                <img
                  src={`${API_BASE_URL}/explainability/${encodeURIComponent(
                    result.explainability.image_path
                      ?.split("\\")
                      .pop()
                      ?.split("/")
                      .pop()
                  )}`}
                  alt="Tumor localization explainability"
                  style={{
                    width: "100%",
                    maxWidth: "900px",
                    display: "block",
                    margin: "20px auto",
                    borderRadius: "10px",
                  }}
                />

                <p className="description">

                  This visualization is an AI
                  interpretability aid and is not
                  clinical ground truth.

                </p>

              </div>

            )}


            {/* =================================================
                PROBABILITIES
            ================================================= */}

            {result.probabilities && (

              <div className="card">

                <h3>
                  AI Classification Probabilities
                </h3>

                <div className="probabilities">

                  {Object.entries(
                    result.probabilities
                  ).map(
                    ([name, probability]) => (

                      <div
                        className="probability-row"
                        key={name}
                      >

                        <div className="probability-header">

                          <span>
                            {name}
                          </span>

                          <strong>

                            {(
                              Number(probability) * 100
                            ).toFixed(2)}%

                          </strong>

                        </div>

                        <div className="progress">

                          <div
                            className="progress-fill"
                            style={{
                              width: `${Number(
                                probability
                              ) * 100}%`,
                            }}
                          />

                        </div>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}


            {/* =================================================
                GUIDANCE
            ================================================= */}

            {(result.guidance ||
              result.recommendation) && (

              <div className="guidance-card">

                <h3>
                  AI-Assisted Finding
                </h3>

                <p>

                  {result.guidance ||
                    result.recommendation}

                </p>

              </div>

            )}


            {/* =================================================
                REPORT
            ================================================= */}

            {result.report && (

              <div className="card report-card">

                <h3>
                  Patient Report
                </h3>

                <p>

                  Your Notebook 6-style
                  AI-assisted report has been
                  generated.

                </p>

                <button
                  className="report-button"
                  onClick={downloadReport}
                  disabled={downloading}
                >

                  {downloading
                    ? "Downloading PDF..."
                    : "Download PDF Report"}

                </button>

              </div>

            )}


            {/* =================================================
                DISCLAIMER
            ================================================= */}

            <div className="disclaimer">

              <strong>
                Important Disclaimer
              </strong>

              <p>

                This system is a research prototype
                for AI-assisted analysis and clinical
                decision support. It is not a medical
                diagnosis and must not replace
                evaluation by a qualified radiologist
                or treating physician.

              </p>

            </div>

          </section>

        )}

      </main>


      {/* =================================================
          FOOTER
      ================================================= */}

      <footer className="footer">

        BrainTumorEdgeAI · Research Prototype ·
        Edge AI Clinical Decision Support

      </footer>

    </div>
  );
}

export default App;