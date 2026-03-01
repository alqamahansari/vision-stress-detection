import React, { useState, useRef } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const API_URL = "https://scaling-pancake-g4xjq475jx65cpg5v-8000.app.github.dev";

function App() {
  const [mode, setMode] = useState(null);
  const [emotion, setEmotion] = useState("");
  const [stress, setStress] = useState(0);
  const [probs, setProbs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const resetResults = () => {
    setEmotion("");
    setStress(0);
    setProbs([]);
    setError("");
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => {
        track.stop();
      });
      videoRef.current.srcObject = null;
    }
  };

  const sendImageToBackend = async (file) => {
    try {
      setLoading(true);
      setError("");

      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API_URL}/predict`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setEmotion(response.data.emotion);
      setStress(response.data.stress_score);
      setProbs(response.data.probabilities);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) sendImageToBackend(file);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (err) {
      console.error(err);
      setError("Camera permission denied.");
    }
  };

  const captureFrame = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      sendImageToBackend(blob);
      stopCamera(); // stop camera after capture
    }, "image/jpeg");
  };

  const stressLevel =
    stress < 0.3 ? "Low" : stress < 0.6 ? "Moderate" : "High";

  const stressColor =
    stress < 0.3 ? "#22c55e" : stress < 0.6 ? "#f59e0b" : "#ef4444";

  const data = {
    labels: [
      "Angry",
      "Disgust",
      "Fear",
      "Happy",
      "Neutral",
      "Sad",
      "Surprise",
    ],
    datasets: [
      {
        label: "Probability",
        data: probs,
      },
    ],
  };

  return (
    <div className="app">
      <div className="title-section">
        <h1>🧠 Workplace Stress Intelligence</h1>
        <p className="subtitle">
          An Explainable Deep Learning Framework for Stress Estimation
        </p>
      </div>

      {mode === null && (
        <div className="mode-selection">
          <div className="mode-card" onClick={() => setMode("camera")}>
            <h2>📷 Live Camera</h2>
            <p>Real-time stress detection</p>
          </div>

          <div className="mode-card" onClick={() => setMode("upload")}>
            <h2>🖼️ Upload Image</h2>
            <p>Analyze a static photo</p>
          </div>
        </div>
      )}

      {mode === "upload" && (
        <div className="feature-container">
          <div className="card">
            <h2>Upload Image</h2>
            <input type="file" onChange={handleUpload} />
          </div>

          <button
            className="back-btn"
            onClick={() => {
              resetResults();
              setMode(null);
            }}
          >
            ← Back
          </button>
        </div>
      )}

      {mode === "camera" && (
        <div className="feature-container">
          <div className="card">
            <h2>Live Camera</h2>

            <video ref={videoRef} autoPlay className="video" />
            <canvas ref={canvasRef} style={{ display: "none" }} />

            <div className="btn-group">
              <button className="primary-btn" onClick={startCamera}>
                Start Camera
              </button>

              <button className="secondary-btn" onClick={captureFrame}>
                Capture & Analyze
              </button>
            </div>
          </div>

          <button
            className="back-btn"
            onClick={() => {
              stopCamera();
              resetResults();
              setMode(null);
            }}
          >
            ← Back
          </button>
        </div>
      )}

      {loading && <p>Analyzing image...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {emotion && !loading && (
        <div className="result-section">
          <div className="result-card">
            <h2>Predicted Emotion: {emotion}</h2>

            <h3 style={{ color: stressColor }}>
              Stress Level: {stressLevel}
            </h3>

            <div className="stress-bar">
              <div
                className="stress-fill"
                style={{
                  width: `${stress * 100}%`,
                  background: stressColor,
                }}
              ></div>
            </div>
          </div>

          <div className="chart-card">
            <Bar data={data} />
          </div>
        </div>
      )}
      <footer className="footer">
        Developed using ResNet50 + Transfer Learning + Face Detection
        </footer>
    </div>
  );
}

export default App;