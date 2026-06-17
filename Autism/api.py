import json
import os
import pickle
from datetime import date, datetime

import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

from .models import User, Case, TestResult, GameScore
from . import db

api = Blueprint('api', __name__)

QCHAT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'autism_model.sav')
FACE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml', 'densenet121_model.keras')

QCHAT_MODEL = None
FACE_MODEL = None
_models_loaded = False

if os.path.exists(QCHAT_MODEL_PATH):
    with open(QCHAT_MODEL_PATH, 'rb') as f:
        QCHAT_MODEL = pickle.load(f)


def _ensure_face_model():
    global FACE_MODEL, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True
    if os.path.exists(FACE_MODEL_PATH):
        FACE_MODEL = load_model(FACE_MODEL_PATH)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
FACE_CASCADE_ALT = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

MIN_FACE_AREA_RATIO = 0.08
MAX_CHILD_AGE_MONTHS = 36

AGE_BUCKETS_MONTHS = ['0-6', '6-12', '12-18', '18-24', '24-30', '30-36', '36+']

ETHNICITY_MAP = {
    'white-european': 0, 'asian': 1, 'middle eastern': 2, 'black': 3,
    'south asian': 4, 'hispanic': 5, 'latino': 6, 'others': 7,
    'mixed': 8, 'pacifica': 9, 'native indian': 10,
}


def calculate_age_months(dob_str, reference=None):
    try:
        dob = datetime.strptime((dob_str or '').strip(), '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None
    if isinstance(reference, datetime):
        reference = reference.date()
    ref = reference or date.today()
    months = (ref.year - dob.year) * 12 + (ref.month - dob.month)
    if ref.day < dob.day:
        months -= 1
    return months if months >= 0 else None


def age_bucket_label(months):
    if months is None:
        return None
    if months < 6:
        return '0-6'
    if months < 12:
        return '6-12'
    if months < 18:
        return '12-18'
    if months < 24:
        return '18-24'
    if months < 30:
        return '24-30'
    if months < 36:
        return '30-36'
    return '36+'

QCHAT_QUESTIONS = [
    'Does your child look at you when you call his/her name?',
    'How easy is it for you to get eye contact with your child?',
    'Does your child point to indicate that s/he wants something?',
    'Does your child point to share interest with you?',
    'Does your child pretend?',
    'Does your child follow where you’re looking?',
    'If someone is upset, does your child try to comfort them?',
    'Would you describe your child’s first words as:',
    'Does your child use simple gestures?',
    'Does your child stare at nothing with no apparent purpose?',
]


def case_payload(case):
    return {
        'id': case.id,
        'child_name': case.child_name,
        'child_dob': case.child_dob,
        'brief': case.brief,
        'created_at': case.created_at.isoformat() if case.created_at else None,
        'last_result_summary': case.last_result_summary,
        'owner_id': case.owner_id,
    }


def result_payload(result):
    return {
        'id': result.id,
        'case_id': result.case_id,
        'created_at': result.created_at.isoformat() if result.created_at else None,
        'spark_score': result.spark_score,
        'image_score': result.image_score,
        'combined_risk': result.combined_risk,
        'prediction_label': result.prediction_label,
        'notes': result.notes,
        'answers': json.loads(result.answers_json or '{}'),
        'report_text': result.report_text,
    }


def require_owner_or_admin(case):
    return current_user.is_admin or case.owner_id == current_user.id


def score_qchat(selected_options):
    scores = []
    for idx, selected in enumerate(selected_options, start=1):
        selected = (selected or '').upper()
        if idx <= 9:
            scores.append(1 if selected in {'C', 'D', 'E'} else 0)
        else:
            scores.append(1 if selected in {'A', 'B', 'C'} else 0)
    return scores


def build_features(data):
    selected_options = [data.get(f'A{i}', '') for i in range(1, 11)]
    scored_answers = score_qchat(selected_options)
    child_sex = 1 if (data.get('child_sex') or '').lower() == 'male' else 0
    ethnicity = ETHNICITY_MAP.get((data.get('child_ethnicity') or '').lower(), 7)
    jaundice = 1 if (data.get('jaundice') or '').lower() == 'yes' else 0
    family_asd = 1 if (data.get('family_asd') or '').lower() == 'yes' else 0
    features = scored_answers + [child_sex, ethnicity, jaundice, family_asd]
    return selected_options, scored_answers, features


def predict_qchat(features):
    if QCHAT_MODEL is None:
        return None, None
    pred = int(QCHAT_MODEL.predict([features])[0])
    prob = None
    if hasattr(QCHAT_MODEL, 'predict_proba'):
        prob = float(QCHAT_MODEL.predict_proba([features])[0][1])
    return pred, prob


def _detect_faces(gray):
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        faces = FACE_CASCADE_ALT.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=6, minSize=(80, 80)
        )
    return faces


def detect_and_crop_face(file_storage):
    file_storage.stream.seek(0)
    file_bytes = file_storage.read()
    file_storage.stream.seek(0)

    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError('Invalid image file.')

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = _detect_faces(gray_eq)
    if len(faces) == 0:
        raise ValueError('No face detected. Please upload a clear front-face image.')

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    if (w * h) / float(img_w * img_h) < MIN_FACE_AREA_RATIO:
        raise ValueError('The detected face is too small. Please upload a closer, clearer photo of the face.')

    eyes = EYE_CASCADE.detectMultiScale(
        gray_eq[y:y + h, x:x + w], scaleFactor=1.05, minNeighbors=6, minSize=(15, 15)
    )
    if len(eyes) == 0:
        raise ValueError('Could not confirm facial features. Please upload a clear, front-facing, well-lit photo.')

    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face


def preprocess_face_for_model(file_storage):
    face = detect_and_crop_face(file_storage)
    face = face.astype('float32')
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face


def predict_face_risk(file_storage):
    _ensure_face_model()
    if FACE_MODEL is None:
        return None

    x = preprocess_face_for_model(file_storage)
    pred = FACE_MODEL.predict(x, verbose=0)

    risk = float(pred[0][0])
    return max(0.0, min(1.0, risk))


def build_report(case, spark_score, qchat_probability, image_probability, combined_probability, selected_options, extra):
    final_prediction = 1 if combined_probability >= 0.5 else 0
    label = 'Autism likelihood detected' if final_prediction == 1 else 'Low autism likelihood'

    qchat_text = f'{(qchat_probability or 0) * 100:.1f}%'
    image_text = f'{(image_probability or 0) * 100:.1f}%'
    final_text = f'{combined_probability * 100:.1f}%'

    lines = [
        f'Auto-Ism Final Report for {case.child_name}',
        f'Case ID: {case.id}',
        '',
        f'Q-CHAT score: {spark_score}/10',
        f'Q-CHAT risk: {qchat_text}',
        f'Facial image risk: {image_text}',
        f'Final weighted risk: {final_text}',
        f'Final assessment: {label}',
        '',
        'Weighting used:',
        'Q-CHAT weight = 0.6',
        'Facial image weight = 0.4',
        '',
        'Question responses:'
    ]

    for idx, ans in enumerate(selected_options, start=1):
        lines.append(f'{idx}. {QCHAT_QUESTIONS[idx - 1]} -> {ans or "No answer"}')

    lines.extend([
        '',
        f"Child sex: {extra.get('child_sex', '')}",
        f"Ethnicity: {extra.get('child_ethnicity', '')}",
        f"Jaundice history: {extra.get('jaundice', '')}",
        f"Family ASD history: {extra.get('family_asd', '')}",
        '',
        'Note: This report is a screening aid and does not replace a clinical diagnosis.'
    ])

    return '\n'.join(lines), label, final_text


@api.route('/questions', methods=['GET'])
def questions():
    return jsonify({
        'questions': [
            {'id': i + 1, 'text': q, 'options': ['A', 'B', 'C', 'D', 'E']}
            for i, q in enumerate(QCHAT_QUESTIONS)
        ],
        'ethnicities': list(ETHNICITY_MAP.keys()),
    })


def age_restriction_error(age_months):
    return (
        f'This assessment is only available for children up to {MAX_CHILD_AGE_MONTHS} months old. '
        f'This child is {age_months} months old.'
    )


@api.route('/validate-face-image', methods=['POST'])
@login_required
def validate_face_image():
    image_file = request.files.get('image')
    case_id = request.form.get('case_id')

    if not image_file:
        return jsonify({'valid': False, 'error': 'No image uploaded.'}), 400

    try:
        _ = detect_and_crop_face(image_file)
        return jsonify({
            'valid': True,
            'message': 'Valid front-face image detected.'
        }), 200
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 400


@api.route('/cases', methods=['GET', 'POST'])
@login_required
def cases():
    if request.method == 'GET':
        query = Case.query.order_by(Case.created_at.desc())
        if not current_user.is_admin:
            query = query.filter_by(owner_id=current_user.id)
        return jsonify({'cases': [case_payload(c) for c in query.all()]})

    data = request.get_json(silent=True) or {}
    child_name = (data.get('child_name') or '').strip()
    child_dob = (data.get('child_dob') or '').strip()
    brief = (data.get('brief') or '').strip()
    if not child_name or not child_dob or not brief:
        return jsonify({'error': 'Please fill all case fields.'}), 400

    age_months = calculate_age_months(child_dob)
    if age_months is not None and age_months > MAX_CHILD_AGE_MONTHS:
        return jsonify({'error': age_restriction_error(age_months)}), 400

    new_case = Case(
        child_name=child_name,
        child_dob=child_dob,
        brief=brief,
        owner_id=current_user.id
    )
    db.session.add(new_case)
    db.session.commit()
    return jsonify({'message': 'Case created successfully.', 'case': case_payload(new_case)})


@api.route('/cases/<int:case_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def case_detail(case_id):
    case = Case.query.get_or_404(case_id)
    if not require_owner_or_admin(case):
        return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'GET':
        return jsonify({'case': case_payload(case)})

    if request.method == 'PUT':
        data = request.get_json(silent=True) or {}
        case.child_name = (data.get('child_name') or case.child_name).strip()
        case.child_dob = (data.get('child_dob') or case.child_dob).strip()
        case.brief = (data.get('brief') or case.brief).strip()
        db.session.commit()
        return jsonify({'message': 'Case updated successfully.', 'case': case_payload(case)})

    db.session.delete(case)
    db.session.commit()
    return jsonify({'message': 'Case deleted.'})


@api.route('/cases/<int:case_id>/test', methods=['POST'])
@login_required
def test_case(case_id):
    case = Case.query.get_or_404(case_id)
    if not require_owner_or_admin(case):
        return jsonify({'error': 'Forbidden'}), 403

    if request.content_type and 'multipart/form-data' in request.content_type:
        data = request.form.to_dict()
        image_file = request.files.get('image')
    else:
        data = request.get_json(silent=True) or {}
        image_file = None

    selected_options, scored_answers, features = build_features(data)
    spark_score = sum(scored_answers)

    qchat_prediction, qchat_probability = predict_qchat(features)

    if qchat_probability is None:
        qchat_probability = spark_score / 10.0

    image_probability = 0.0
    if image_file:
        try:
            image_probability = predict_face_risk(image_file)
            if image_probability is None:
                image_probability = 0.0
        except Exception as e:
            return jsonify({'error': f'Image model failed: {str(e)}'}), 400

    combined_probability = (0.6 * qchat_probability) + (0.4 * image_probability)
    final_prediction = 1 if combined_probability >= 0.5 else 0

    extra = {
        'child_sex': data.get('child_sex', ''),
        'child_ethnicity': data.get('child_ethnicity', ''),
        'jaundice': data.get('jaundice', ''),
        'family_asd': data.get('family_asd', ''),
    }

    report_text, label, risk_text = build_report(
        case,
        spark_score,
        qchat_probability,
        image_probability,
        combined_probability,
        selected_options,
        extra
    )

    answers = {f'A{i}': selected_options[i - 1] for i in range(1, 11)}
    answers.update(extra)

    result = TestResult(
        case_id=case.id,
        spark_score=spark_score,
        image_score=image_probability,
        combined_risk=combined_probability,
        prediction_label=label,
        notes=(
            f'Q-CHAT risk: {qchat_probability * 100:.1f}% | '
            f'Facial risk: {image_probability * 100:.1f}% | '
            f'Final risk: {combined_probability * 100:.1f}%'
        ),
        answers_json=json.dumps(answers),
        report_text=report_text,
    )

    db.session.add(result)
    case.last_result_summary = f'Final risk {risk_text} | Q-CHAT {spark_score}/10'
    db.session.commit()

    return jsonify({
        'message': 'Assessment submitted successfully.',
        'result': result_payload(result)
    })


@api.route('/cases/<int:case_id>/reports', methods=['GET'])
@login_required
def case_reports(case_id):
    case = Case.query.get_or_404(case_id)
    if not require_owner_or_admin(case):
        return jsonify({'error': 'Forbidden'}), 403

    results = TestResult.query.filter_by(case_id=case.id).order_by(TestResult.created_at.desc()).all()
    return jsonify({'reports': [result_payload(r) for r in results]})


@api.route('/reports/<int:report_id>', methods=['GET'])
@login_required
def report_detail(report_id):
    result = TestResult.query.get_or_404(report_id)
    if not require_owner_or_admin(result.case):
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify({'report': result_payload(result), 'case': case_payload(result.case)})


GAME_TYPES = {'recognition', 'shapes', 'emotions'}


def game_score_payload(gs):
    return {
        'id': gs.id,
        'case_id': gs.case_id,
        'game': gs.game,
        'level': gs.level,
        'score': gs.score,
        'max_score': gs.max_score,
        'updated_at': gs.updated_at.isoformat() if gs.updated_at else None,
    }


@api.route('/cases/<int:case_id>/game-scores', methods=['GET', 'POST'])
@login_required
def case_game_scores(case_id):
    case = Case.query.get_or_404(case_id)
    if not require_owner_or_admin(case):
        return jsonify({'error': 'Forbidden'}), 403

    if request.method == 'GET':
        scores = GameScore.query.filter_by(case_id=case.id).all()
        return jsonify({'game_scores': [game_score_payload(s) for s in scores]})

    data = request.get_json(silent=True) or {}
    game = (data.get('game') or '').strip().lower()
    if game not in GAME_TYPES:
        return jsonify({'error': 'Invalid game type.'}), 400
    try:
        score = int(data.get('score', 0))
        max_score = int(data.get('max_score', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid score values.'}), 400
    level = (data.get('level') or '').strip()

    existing = GameScore.query.filter_by(case_id=case.id, game=game).first()
    if existing:
        existing.level = level
        existing.score = score
        existing.max_score = max_score
        existing.updated_at = datetime.utcnow()
    else:
        existing = GameScore(case_id=case.id, game=game, level=level, score=score, max_score=max_score)
        db.session.add(existing)
    db.session.commit()
    return jsonify({'message': 'Score saved.', 'game_score': game_score_payload(existing)})


@api.route('/admin/dashboard', methods=['GET'])
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        return jsonify({'error': 'Forbidden'}), 403

    total_accounts = User.query.count()
    total_cases = Case.query.count()
    total_tests = TestResult.query.count()
    recent_reports = TestResult.query.order_by(TestResult.created_at.desc()).limit(10).all()
    risk_average = db.session.query(func.avg(TestResult.combined_risk)).scalar() or 0

    positive_tests = TestResult.query.filter(TestResult.combined_risk >= 0.5).count()
    negative_tests = total_tests - positive_tests

    age_distribution = {b: 0 for b in AGE_BUCKETS_MONTHS}
    for case in Case.query.all():
        bucket = age_bucket_label(calculate_age_months(case.child_dob))
        if bucket:
            age_distribution[bucket] += 1

    age_vs_result = {b: {'positive': 0, 'negative': 0} for b in AGE_BUCKETS_MONTHS}
    sex_vs_result = {'male': {'positive': 0, 'negative': 0}, 'female': {'positive': 0, 'negative': 0}}
    ethnicity_vs_result = {key: {'positive': 0, 'negative': 0} for key in ETHNICITY_MAP.keys()}

    for result in TestResult.query.all():
        outcome = 'positive' if (result.combined_risk or 0) >= 0.5 else 'negative'

        case = result.case
        if case:
            bucket = age_bucket_label(calculate_age_months(case.child_dob, result.created_at))
            if bucket:
                age_vs_result[bucket][outcome] += 1

        try:
            answers = json.loads(result.answers_json or '{}')
        except ValueError:
            answers = {}

        sex = (answers.get('child_sex') or '').lower()
        if sex in sex_vs_result:
            sex_vs_result[sex][outcome] += 1

        ethnicity = (answers.get('child_ethnicity') or '').lower()
        if ethnicity in ethnicity_vs_result:
            ethnicity_vs_result[ethnicity][outcome] += 1

    return jsonify({
        'stats': {
            'total_accounts': total_accounts,
            'total_cases': total_cases,
            'total_tests': total_tests,
            'average_risk': round(float(risk_average or 0) * 100, 2),
            'positive_tests': positive_tests,
            'negative_tests': negative_tests,
        },
        'age_distribution': age_distribution,
        'age_vs_result': age_vs_result,
        'sex_vs_result': sex_vs_result,
        'ethnicity_vs_result': ethnicity_vs_result,
        'recent_reports': [result_payload(r) for r in recent_reports],
    })


@api.route('/admin/users', methods=['GET'])
@login_required
def admin_users():
    if not current_user.is_admin:
        return jsonify({'error': 'Forbidden'}), 403

    users = User.query.order_by(User.created_at.desc()).all()
    payload = []
    for u in users:
        payload.append({
            'id': u.id,
            'email': u.email,
            'firstname': u.firstname,
            'lastname': u.lastname,
            'is_admin': u.is_admin,
            'created_at': u.created_at.isoformat() if u.created_at else None,
            'cases_count': len(u.cases),
        })
    return jsonify({'users': payload})


@api.route('/admin/users/<int:user_id>', methods=['DELETE'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Forbidden'}), 403

    user = User.query.get_or_404(user_id)
    if user.is_admin:
        return jsonify({'error': 'Admin account cannot be deleted.'}), 400

    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully.'})