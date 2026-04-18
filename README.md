# ⚽ GameMind – Football AI Analysis Platform

GameMind is an AI-powered football analysis system that processes match footage to extract meaningful insights such as player tracking and event detection.

The system combines computer vision and deep learning techniques with a modern web interface to deliver fast and interactive analysis.

---

## 🚀 Features

### 🎯 Player Tracking

* Detects and tracks players, referees, and the ball
* Assigns teams based on jersey color
* Calculates player speed and distance covered
* Displays annotated video output

### ⚡ Event Detection

* Detects key match events such as:

  * Goals
  * Shots
  * Passes
  * Fouls
* Provides timestamps and confidence scores

### 🌐 Web Interface

* Upload match videos
* Choose between tracking or event detection
* Visualize results instantly in browser
* Modern UI built with React

---

## 🏗️ Project Structure

```
GameMind/
│
├── backend/                # Flask backend (AI processing)
│   ├── app.py
│   ├── tracking_pipeline.py
│   ├── test_pipeline.py
│   ├── tracking/
│   └── utils/
│
├── frontend/               # React frontend
│   ├── src/
│   ├── public/
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 🔹 Backend (Flask)

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
```

---

### 🔹 Frontend (React)

```bash
cd frontend
npm install
npm start
```

---

## ▶️ Usage

1. Open the web app in your browser
2. Upload a football match video
3. Choose:

   * Player Tracking
   * Event Detection
4. Click **Run Analysis**
5. View results:

   * Annotated video
   * Detected events

---

## 🛠️ Technologies Used

* Python (Flask)
* OpenCV
* YOLO (Object Detection)
* NumPy
* React.js
* Axios

---

## 📌 Notes

* Make sure FFmpeg is installed for video processing
* Large videos may take time to process depending on hardware

---

## ⭐ Future Improvements

* Real-time match analysis
* Tactical heatmaps
* Player performance dashboards
* Cloud deployment

---

## 📄 License

This project is for educational and research purposes.
