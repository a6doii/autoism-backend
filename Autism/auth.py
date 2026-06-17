import os
import time

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from .models import User
from . import db, generate_token

auth = Blueprint('auth', __name__)

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def user_payload(user):
    return {
        'id': user.id,
        'email': user.email,
        'firstname': user.firstname,
        'lastname': user.lastname,
        'dob': user.dob,
        'sex': user.sex,
        'profile_image': user.profile_image,
        'is_admin': user.is_admin,
    }


@auth.route('/me', methods=['GET'])
def me():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'user': user_payload(current_user)})
    return jsonify({'authenticated': False, 'user': None})


@auth.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or request.form
    firstname = (data.get('firstName') or data.get('firstname') or '').strip()
    lastname = (data.get('lastName') or data.get('lastname') or '').strip()
    email = (data.get('email') or '').strip()
    dob = (data.get('dob') or data.get('dateOfBirth') or '').strip()
    sex = (data.get('sex') or '').strip()
    password = data.get('password') or ''
    confirm = data.get('confirm') or data.get('confirmPassword') or password

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email is already registered.'}), 400
    if len(firstname) < 2 or len(lastname) < 2:
        return jsonify({'error': 'Please enter a valid first and last name.'}), 400
    if len(email) < 3:
        return jsonify({'error': 'Please enter a valid email.'}), 400
    if password != confirm:
        return jsonify({'error': 'Passwords do not match.'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters.'}), 400

    user = User(
        firstname=firstname,
        lastname=lastname,
        email=email,
        dob=dob,
        sex=sex,
        password=generate_password_hash(password, method='pbkdf2:sha256', salt_length=16),
    )
    db.session.add(user)
    db.session.commit()
    login_user(user, remember=True)
    return jsonify({'message': 'Account created successfully.', 'user': user_payload(user), 'token': generate_token(user.id)})


@auth.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or request.form
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid email or password.'}), 401
    login_user(user, remember=True)
    return jsonify({'message': 'Logged in successfully.', 'user': user_payload(user), 'token': generate_token(user.id)})


@auth.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully.'})


@auth.route('/profile', methods=['GET', 'PUT'])
@login_required
def profile():
    if request.method == 'GET':
        return jsonify({'user': user_payload(current_user)})

    data = request.get_json(silent=True) or {}
    current_user.firstname = (data.get('firstName') or current_user.firstname).strip()
    current_user.lastname = (data.get('lastName') or current_user.lastname).strip()
    current_user.dob = (data.get('dob') or data.get('dateOfBirth') or current_user.dob or '').strip()
    current_user.sex = (data.get('sex') or current_user.sex or '').strip()
    new_email = (data.get('email') or current_user.email).strip()
    existing = User.query.filter(User.email == new_email, User.id != current_user.id).first()
    if existing:
        return jsonify({'error': 'Email is already used by another account.'}), 400
    current_user.email = new_email
    db.session.commit()
    return jsonify({'message': 'Profile updated.', 'user': user_payload(current_user)})


@auth.route('/profile/picture', methods=['POST'])
@login_required
def upload_profile_picture():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({'error': 'No image uploaded.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'error': 'Unsupported image format. Use JPG, PNG, or WEBP.'}), 400

    upload_dir = current_app.config['UPLOAD_FOLDER']

    if current_user.profile_image:
        old_path = os.path.join(upload_dir, os.path.basename(current_user.profile_image))
        if os.path.exists(old_path):
            os.remove(old_path)

    filename = f'user_{current_user.id}_{int(time.time())}{ext}'
    file.save(os.path.join(upload_dir, filename))

    current_user.profile_image = f'/static/uploads/{filename}'
    db.session.commit()
    return jsonify({'message': 'Profile picture updated.', 'user': user_payload(current_user)})
