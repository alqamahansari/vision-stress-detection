# 🧠 An Explainable Computer Vision-Based Framework for Workplace Stress Estimation Using Deep Learning

A research-driven deep learning system that estimates workplace stress levels from facial expressions using Transfer Learning on ResNet50 combined with face detection and interpretable stress scoring logic.


## 📌 Project Overview

This capstone project presents a computer vision framework for automatic stress estimation from facial expressions.

The system performs:

- Face Detection using OpenCV Haar Cascade
- Emotion Classification using a fine-tuned ResNet50 model
- Softmax probability estimation
- Weighted stress score computation
- Stress level categorization (Low / Moderate / High)
- Web-based visualization using Flask

The project integrates deep learning with a lightweight web deployment architecture.


## 🧠 Model Architecture

- Base Model: ResNet50
- Strategy: Transfer Learning
- Framework: PyTorch
- Output Classes (7):
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise


## 📊 Stress Score Calculation

Stress is computed using weighted emotional probabilities:

Stress Score = (0.5 × Angry) + (0.3 × Sad) + (0.2 × Fear)

Stress Levels:
- Low: < 0.33
- Moderate: 0.33 – 0.66
- High: > 0.66

This provides an interpretable and explainable stress estimation mechanism.

## 🖥️ Features

- Multi-page Flask Web Application
- Image Upload Stress Analysis
- Live Camera Stress Analysis
- Automatic Face Detection
- Emotion Confidence Percentage
- Stress Score Visualization Bar
- Probability Distribution Output
- Clean Research-Oriented UI
- Hugging Face Model Hosting Integration
- Docker Support (Optional)


## 📂 Project Structure
``````
vision-stress-detection/
│
├── backend/
│ ├── app.py
│ └── requirements.txt
│
├── frontend/
│ └── src/
│   └── App.js
│
├── .gitignore
└── requirements.txt
``````

## 🚀 Installation

### 1️⃣ Clone Repository
```bash
https://github.com/alqamahansari/vision-stress-detection.git
cd vision-stress-detection
```


### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```


### 3️⃣ Run Application
```bash
python app.py
```

Open in browser:
``````
http://127.0.0.1:5000
``````


## 🌐 Model Hosting

The trained model is hosted on Hugging Face:

``````
Alquamah/emotion-stress-resnet50
``````


The model is automatically downloaded during application startup.


## 📊 Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- OpenCV
- Hugging Face Hub
- React JS


## 🎓 Academic Contribution

This project demonstrates:

- Application of transfer learning in affective computing
- Interpretable stress estimation logic
- Integration of deep learning into practical web systems
- End-to-end AI deployment pipeline

It serves as a foundation for future research in:

- Explainable AI for mental health analytics
- Workplace behavioral analytics
- Vision-based psychological state estimation

## ⚠️ Disclaimer

This system is intended for academic research purposes only.  
It is not a clinical diagnostic tool and should not be used for medical or psychological decisions.


## 👨‍💻 Author

Mohammad Alquamah Ansari  
B.Sc. Artificial Intelligence  
ML Engineer | DL & LLMs


## 📜 License

This project is developed for academic and research purposes.