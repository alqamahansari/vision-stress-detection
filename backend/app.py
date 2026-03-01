from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from huggingface_hub import hf_hub_download
import cv2
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Downloading model from Hugging Face...")

model_path = hf_hub_download(
    repo_id="Alquamah/emotion-stress-resnet50",
    filename="emotion_model_finetuned.pth"
)

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 7)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load face detector once (not every request)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running successfully."})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image_bytes = file.read()

        # Convert to PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert to OpenCV
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({
                "emotion": "No face detected",
                "confidence": 0.0,
                "stress_score": 0.0,
                "stress_level": "Unknown"
            })

        # Crop first detected face
        x, y, w, h = faces[0]
        face = open_cv_image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(face)

        # Match FER style
        image = image.convert("L").convert("RGB")

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        predicted_index = int(np.argmax(probs))
        predicted_class = classes[predicted_index]
        confidence = float(np.max(probs))

        # Confidence threshold
        if confidence < 0.5:
            return jsonify({
                "emotion": "Uncertain",
                "confidence": confidence,
                "stress_score": 0.0,
                "stress_level": "Uncertain",
                "probabilities": probs.tolist()
            })

        angry = probs[0]
        fear = probs[2]
        sad = probs[5]

        stress_score = (
            0.5 * angry +
            0.3 * sad +
            0.2 * fear
        )

        # Clamp between 0 and 1
        stress_score = float(np.clip(stress_score, 0, 1))

        if stress_score < 0.33:
            stress_level = "Low"
        elif stress_score < 0.66:
            stress_level = "Moderate"
        else:
            stress_level = "High"

        return jsonify({
            "emotion": predicted_class,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "stress_score": stress_score,
            "stress_level": stress_level
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)