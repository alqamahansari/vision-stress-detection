from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from model import EmotionResNet

app = Flask(__name__)
CORS(app)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = EmotionResNet().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_class = classes[np.argmax(probs)]

    angry = probs[0]
    fear = probs[2]
    sad = probs[5]

    stress_score = 0.4 * angry + 0.4 * sad + 0.2 * fear

    return jsonify({
        "emotion": predicted_class,
        "probabilities": probs.tolist(),
        "stress_score": float(stress_score)
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)