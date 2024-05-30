from flask import Blueprint, request, jsonify
from models import db, User
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity,get_jwt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from io import BytesIO
import requests

auth_bp = Blueprint('auth', __name__)

# Load the trained model once when the application starts
model = load_model('skin_cancer_detection_model.keras')

# Define the class names mapping
class_names = {
    0: 'Actinic Keratoses Intraepithelial Carcinomas (AKIEC',
    1: 'Basal cell carcinoma (BCC)',
    2: 'Keratinocytic Lesions (BKL)',
    3: 'Dermatofibromas (DF)',
    4: 'Melanoma',
    5: 'Nevus',
    6: 'Vascular Tumors '
}

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({"message": "User already exists"}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity=user.id)
        return jsonify(message="Login successful", access_token=access_token), 200

    return jsonify({"message": "Invalid credentials"}), 401

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    return jsonify({"message": "Logout successful"}), 200

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    return jsonify(username=user.username, email=user.email), 200

@auth_bp.route('/detect', methods=['POST'])
@jwt_required()
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the image (resize and normalize)
        target_size = (224, 224)
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = preprocess_input(img)  # Normalization specific to EfficientNet

        # Expand dimensions to match model input
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map predicted class to label
        result = class_names.get(predicted_class, 'Unknown')

        return jsonify({'prediction': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@auth_bp.route('/detect-url', methods=['POST'])
@jwt_required()
def detect_url():
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not (url.startswith('http://') or url.startswith('https://')):
            return jsonify({'error': 'Invalid URL format. Please enter a URL starting with http:// or https://'}), 400

        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({'error': 'Unable to fetch image from URL'}), 400

        img = Image.open(BytesIO(response.content)).convert('RGB')

        # Preprocess the image (resize and normalize)
        img = np.array(img)
        target_size = (224, 224)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32)
        img = preprocess_input(img)  # Normalization specific to EfficientNet

        # Expand dimensions to match model input
        img = np.expand_dims(img, axis=0)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map predicted class to label
        result = class_names.get(predicted_class, 'Unknown')

        return jsonify({'prediction': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500