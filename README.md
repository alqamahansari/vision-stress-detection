# 🧠 Vision-Based Workplace Stress Intelligence System

### An Explainable Computer Vision-Based Framework for Workplace Stress Estimation Using Deep Learning


## 📌 Overview

This project presents an end-to-end deep learning framework for estimating workplace stress levels from facial expressions using computer vision.

The system integrates:

- 🧠 ResNet50 with Transfer Learning
- 🎯 Facial Emotion Recognition (7 classes)
- 📊 Stress Score Computation Model
- 🔍 Face Detection (Haar Cascade)
- 🌍 Cloud Deployment Architecture
- 📈 Probability Visualization Dashboard

The application is fully containerized and deployed using modern DevOps practices.


## 🏗 System Architecture
``````
Frontend (React - GitHub Pages)
↓
Backend (Flask API - Render/Railway)
↓
ResNet50 Model (Hugging Face)
``````


## 🎯 Problem Statement

Workplace stress significantly impacts productivity, mental health, and organizational performance. This system aims to provide a non-invasive AI-based framework to estimate stress levels using facial emotion signals.


## 🧠 Model Details

- Architecture: ResNet50
- Pretraining: ImageNet
- Fine-tuning: FER-based emotion dataset
- Output Classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

### Stress Score Formula
``````
Stress Score = 0.5 × Angry + 0.3 × Sad + 0.2 × Fear
``````

Stress Levels:

- Low (< 0.33)
- Moderate (0.33 – 0.66)
- High (> 0.66)


## 🔍 Explainability

The system enhances interpretability through:

- Emotion probability distribution charts
- Confidence scores
- Stress scoring transparency
- Structured inference pipeline


## 🚀 Deployment Stack

### Frontend
- React.js
- GitHub Pages

### Backend
- Flask
- Gunicorn
- OpenCV (Headless)
- PyTorch
- Hugging Face Hub

### DevOps
- Dockerized backend
- Cloud deployment (Render)
- CI/CD ready structure


## 📁 Project Structure
``````
vision-stress-detection/
│
├── frontend/ # React UI
│ ├── src/
│ └── package.json
│
├── backend/ # Flask API
│ ├── app.py
│ ├── requirements.txt
│ └── Dockerfile
│
└── README.md
``````

## 💻 Running Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```


Runs at:
``````
http://localhost:8000
``````


### Frontend
```bash
cd frontend
npm install
npm start
```

Runs At:
``````
http://localhost:3000
``````


## 🌍 Production Deployment

### Backend
- Hosted on Render
- Public API endpoint

### Frontend
- Hosted via GitHub Pages


## 🔐 Ethical Considerations

- Designed for academic research and controlled environments.
- Not intended for medical diagnosis.
- Privacy-preserving architecture (no image storage).
- Transparent stress scoring logic.


## 📊 Example Output

- Predicted Emotion: Angry
- Confidence: 78.2%
- Stress Level: High
- Stress Score: 0.71


## 📚 Technologies Used

- Python
- PyTorch
- OpenCV
- Flask
- React.js
- Docker
- Cloud Deployment Platforms


## 🎓 Academic Context

Capstone Project for:

Bachelor of Science in Artificial Intelligence


## 📌 Future Improvements

- Grad-CAM Visualization
- Real-time video stream processing
- Emotion smoothing over time
- Model quantization for faster inference
- Kubernetes-based scaling
- MLOps pipeline integration


## 👨‍💻 Author

Mohammad Alquamah Ansari  
B.Sc. Artificial Intelligence  
ML Engineer | Ethical AI & Explainable Deep Learning

## 📄 License

This project is for academic and research purposes only.
