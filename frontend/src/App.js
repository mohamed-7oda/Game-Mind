import React, { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

const MODES = [
  {
    id: "tracking",
    label: "Player Tracking",
    icon: "⬡",
    desc: "AI-powered positional mapping & heatmaps",
  },
  {
    id: "event",
    label: "Event Detection",
    icon: "◈",
    desc: "Automated tagging of key match moments",
  },
];

const EVENT_COLORS = {
  goal: "#f97316",
  shot: "#facc15",
  pass: "#38bdf8",
  foul: "#f87171",
  default: "#a3e635",
};

function getEventColor(name = "") {
  const lower = name.toLowerCase();
  for (const key of Object.keys(EVENT_COLORS)) {
    if (lower.includes(key)) return EVENT_COLORS[key];
  }
  return EVENT_COLORS.default;
}

function ConfBar({ value }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${pct}%`, background: pct > 80 ? "#a3e635" : pct > 50 ? "#facc15" : "#f87171" }}
        />
      </div>
      <span className="conf-label">{pct}%</span>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("tracking");
  const [result, setResult] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileRef = useRef();

  const handleFile = (f) => {
    if (f && f.type.startsWith("video/")) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return alert("Upload a video first");
    setLoading(true);
    setResult(null);
    setVideoUrl("");
    setProgress(0);

    const formData = new FormData();
    formData.append("video", file);
    formData.append("mode", mode);

    try {
      const response = await axios.post("http://127.0.0.1:5000/process", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: mode === "tracking" ? "blob" : "json",
        onUploadProgress: (e) => setProgress(Math.round((e.loaded / e.total) * 100)),
      });

      if (mode === "tracking") {
        const blob = new Blob([response.data], { type: "video/mp4" });
        const url = URL.createObjectURL(blob);
        setVideoUrl(url);
      } else {
        setResult(response.data);
      }
    } catch {
      alert("Error processing video");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      {/* Pitch grid background */}
      <div className="pitch-bg" aria-hidden />

      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">TACTIQ<span className="logo-sub">AI</span></span>
          </div>
          <nav className="nav">
            <span className="nav-item active">Analysis</span>
            <span className="nav-item">Reports</span>
            <span className="nav-item">Squad</span>
          </nav>
          <div className="header-badge">BETA</div>
        </div>
      </header>

      <main className="main">
        {/* Hero */}
        <section className="hero">
          <div className="hero-eyebrow">Football Intelligence Platform</div>
          <h1 className="hero-title">
            Decode Every<br />
            <span className="hero-accent">Match Moment</span>
          </h1>
          <p className="hero-sub">
            Upload footage. Our AI tracks players, detects events, and surfaces
            tactical insights in seconds.
          </p>
        </section>

        {/* Control Panel */}
        <section className="panel">
          {/* Mode selector */}
          <div className="mode-grid">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`mode-card ${mode === m.id ? "mode-active" : ""}`}
                onClick={() => setMode(m.id)}
              >
                <span className="mode-icon">{m.icon}</span>
                <span className="mode-label">{m.label}</span>
                <span className="mode-desc">{m.desc}</span>
                {mode === m.id && <span className="mode-pip" />}
              </button>
            ))}
          </div>

          {/* Drop zone */}
          <div
            className={`dropzone ${dragOver ? "dz-over" : ""} ${file ? "dz-filled" : ""}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <input
              ref={fileRef}
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {file ? (
              <div className="dz-file">
                <span className="dz-file-icon">▶</span>
                <div>
                  <div className="dz-filename">{file.name}</div>
                  <div className="dz-meta">{(file.size / 1e6).toFixed(1)} MB · {file.type.split("/")[1].toUpperCase()}</div>
                </div>
                <span className="dz-change">Change</span>
              </div>
            ) : (
              <div className="dz-empty">
                <span className="dz-upload-icon">↑</span>
                <span className="dz-cta">Drop match footage here</span>
                <span className="dz-hint">MP4, MOV, AVI · Up to 2 GB</span>
              </div>
            )}
          </div>

          {/* Progress bar (upload) */}
          {loading && (
            <div className="progress-wrap">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="progress-label">{progress < 100 ? `Uploading ${progress}%` : "Processing…"}</span>
            </div>
          )}

          <button className="run-btn" onClick={handleUpload} disabled={loading || !file}>
            {loading ? (
              <><span className="spinner" /> Analysing…</>
            ) : (
              <><span className="run-icon">◈</span> Run Analysis</>
            )}
          </button>
        </section>

        {/* Video result */}
        {videoUrl && (
          <section className="result-section">
            <div className="result-header">
              <span className="result-badge">TRACKING OUTPUT</span>
              <h2 className="result-title">Processed Footage</h2>
            </div>
            <div className="video-wrap">
              <video src={videoUrl} controls />
              <div className="video-overlay-corner top-left">TACTIQ AI</div>
              <div className="video-overlay-corner top-right">LIVE TRACK</div>
            </div>
          </section>
        )}

        {/* Events result */}
        {result && (
          <section className="result-section">
            <div className="result-header">
              <span className="result-badge">EVENT DETECTION</span>
              <h2 className="result-title">
                {result.length} Event{result.length !== 1 ? "s" : ""} Detected
              </h2>
            </div>

            <div className="events-table">
              <div className="events-thead">
                <span>Timestamp</span>
                <span>Event</span>
                <span>Confidence</span>
              </div>
              {result.map((e, i) => (
                <div key={i} className="event-row" style={{ "--accent": getEventColor(e.event) }}>
                  <span className="ev-time">{e.time}</span>
                  <span className="ev-name">
                    <span className="ev-dot" />
                    {e.event}
                  </span>
                  <ConfBar value={e.confidence} />
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        © 2025 TACTIQ AI · Professional Football Intelligence
      </footer>
    </div>
  );
}