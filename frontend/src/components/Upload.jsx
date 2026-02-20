import React, { useState } from "react";
import { predictKidneyTumor } from "../api";

const Upload = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  // Handle prediction
  const handleUpload = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);

    try {
      const data = await predictKidneyTumor(file);
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Check backend.");
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>
        🧠 Kidney Tumor Identification System
      </h1>

      <p style={styles.subtitle}>
        AI-powered CT scan analysis for early tumor detection
      </p>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ marginTop: "20px" }}
      />

      {/* Image Preview */}
      {preview && (
        <div style={{ marginTop: "20px" }}>
          <img src={preview} alt="preview" style={styles.preview} />
        </div>
      )}

      {/* Predict Button */}
      <button
        onClick={handleUpload}
        disabled={loading}
        style={styles.button}
      >
        {loading ? "🔄 Analyzing..." : "🔍 Predict"}
      </button>

      {/* Result */}
      {result && (
        <div style={styles.resultCard}>
          <h3>Prediction: {result.prediction}</h3>
          <p>
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      )}

      {/* Disclaimer */}
      <p style={styles.disclaimer}>
        ⚠️ This AI tool is for research purposes only and not a
        substitute for professional medical diagnosis.
      </p>
    </div>
  );
};

// 🎨 Simple medical styling
const styles = {
  container: {
    textAlign: "center",
    padding: "40px",
    fontFamily: "Arial, sans-serif",
    backgroundColor: "#f4f7fb",
    minHeight: "100vh",
  },
  title: {
    color: "#1f3c88",
  },
  subtitle: {
    color: "#555",
  },
  preview: {
    width: "300px",
    borderRadius: "10px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
  },
  button: {
    marginTop: "20px",
    padding: "12px 24px",
    fontSize: "16px",
    backgroundColor: "#1f3c88",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  },
  resultCard: {
    marginTop: "25px",
    padding: "20px",
    backgroundColor: "white",
    borderRadius: "10px",
    display: "inline-block",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  },
  disclaimer: {
    marginTop: "40px",
    fontSize: "12px",
    color: "#888",
  },
};

export default Upload;
