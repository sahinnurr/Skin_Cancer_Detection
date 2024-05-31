from flask import Blueprint, request, jsonify
from models import db, User
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
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
    0: 'Actinic Keratoses Intraepithelial Carcinomas (AKIEC)',
    1: 'Basal cell carcinoma (BCC)',
    2: 'Keratinocytic Lesions (BKL)',
    3: 'Dermatofibromas (DF)',
    4: 'Melanoma',
    5: 'Nevus',
    6: 'Vascular Tumors'
}

# Define detailed information for each skin cancer type
skin_cancer_info = {
    'Actinic Keratoses Intraepithelial Carcinomas (AKIEC)': 'Actinic keratoses are rough, scaly patches on the skin caused by years of sun exposure. They are often found on the face, lips, ears, forearms, scalp, neck, or back of the hands. '
                                                            'AKIEC is a more severe form and can develop into skin cancer. '
                                                            'Early diagnosis and treatment can prevent progression to squamous cell carcinoma, a more aggressive form of skin cancer. '
                                                            'Treatment options include cryotherapy, topical medications, and photodynamic therapy.',
    'Basal cell carcinoma (BCC)': 'This is the most common type of skin cancer. '
                                  'BCC frequently develops in people who have fair skin. People who have skin of color also get this skin cancer. '
                                  'BCCs often look like a flesh-colored round growth, pearl-like bump, or a pinkish patch of skin. '
                                  'BCCs usually develop after years of frequent sun exposure or indoor tanning. '
                                  'BCCs are common on the head, neck, and arms; however, they can form anywhere on the body, including the chest, abdomen, and legs. '
                                  'Early diagnosis and treatment for BCC are important. BCC can grow deep. Allowed to grow, it can penetrate the nerves and bones, causing damage and disfigurement.',
    'Keratinocytic Lesions (BKL)': 'BKL includes benign keratinocytic lesions such as seborrheic keratosis. These are non-cancerous skin growths that can appear anywhere on the body and are often brown, black, or light tan. '
                                   'Seborrheic keratoses are typically wart-like in appearance and have a waxy, pasted-on-the-skin look. They are common in older adults and can increase in number as one ages. '
                                   'While they are harmless, they can sometimes be itchy or irritated, and removal options include cryotherapy, curettage, or laser therapy for cosmetic reasons.',
    'Dermatofibromas (DF)': 'DF is a rare skin cancer. It begins in the middle layer of skin, the dermis. DFSP tends to grow slowly. It seldom spreads to other parts of the body.'
                            'Because DFSP rarely spreads, this cancer has a high survival rate. Treatment is important, though. Without treatment, DFSP can grow deep into the fat, muscle, and even bone. If this happens, treatment can be difficult.'
                            'The first sign of this skin cancer is often a small bump on the skin. It may resemble a deep-seated pimple or rough patch of skin. DF can also look like a scar. In children, it may remind you of a birthmark.',
    'Melanoma': 'Melanoma is often called "the most serious skin cancer" because it has a tendency to spread.'
                'Melanoma can develop within a mole that you already have on your skin or appear suddenly as a dark spot on the skin that looks different from the rest.'
                'Early diagnosis and treatment are crucial.'
                'Knowing the ABCDE warning signs of melanoma can help you find an early melanoma.',
    'Nevus': 'A nevus is a common benign skin lesion, often referred to as a mole. They can be flat or raised and vary in color from flesh-toned to dark brown or black.'
            'While most nevi are harmless, some can develop into melanoma, a serious form of skin cancer. '
            'Monitoring changes in size, shape, or color of a nevus is important for early detection of melanoma. If a nevus shows signs of asymmetry, border irregularities, color changes, diameter larger than 6mm, or evolving appearance, it should be evaluated by a dermatologist.'
            'Removal of nevi can be done for cosmetic reasons or if there is suspicion of malignancy.',

    'Vascular Tumors': 'Vascular tumors include a variety of benign and malignant growths that originate from blood vessels. These can appear as red or purple marks on the skin and may require medical attention based on their nature.'
                        'Hemangiomas are common benign vascular tumors often seen in infants, which usually regress on their own.'
                        'Angiosarcomas are rare malignant vascular tumors that require aggressive treatment, including surgery, radiation, and chemotherapy.'
                        'Vascular tumors can vary widely in appearance and behavior, making accurate diagnosis and appropriate management crucial.'
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
        info = skin_cancer_info.get(result, 'No additional information available.')

        # Map predicted classes to labels and confidence percentages
        results = []
        for i, prediction in enumerate(predictions[0]):
            class_name = class_names.get(i, 'Unknown')
            confidence_percentage = prediction * 100
            results.append({'class_name': class_name, 'confidence': confidence_percentage})


        return jsonify({'prediction': result, 'information': info, 'predictions': results}), 200

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
        info = skin_cancer_info.get(result, 'No additional information available.')

        # Map predicted classes to labels and confidence percentages
        results = []
        for i, prediction in enumerate(predictions[0]):
            class_name = class_names.get(i, 'Unknown')
            confidence_percentage = prediction * 100
            results.append({'class_name': class_name, 'confidence': confidence_percentage})

        return jsonify({'prediction': result, 'information': info, 'predictions': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

