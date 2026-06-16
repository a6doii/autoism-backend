
from flask import Blueprint, render_template, request, flash, redirect, url_for, abort
from flask_login import login_required, current_user
from os import path
import pickle

from .models import Case, TestResult
from . import db

views = Blueprint("views", __name__)

MODEL_PATH = path.join(path.dirname(__file__), "ml", "autism_model.sav")
QCHAT_MODEL = None

if path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        QCHAT_MODEL = pickle.load(f)

QCHAT_QUESTIONS = [
    {
        "id": 1,
        "text": "Does your child look at you when you call his/her name?",
        "options": [("A", "Always"), ("B", "Usually"), ("C", "Sometimes"), ("D", "Rarely"), ("E", "Never")],
    },
    {
        "id": 2,
        "text": "How easy is it for you to get eye contact with your child?",
        "options": [("A", "Very easy"), ("B", "Quite easy"), ("C", "Quite difficult"), ("D", "Very difficult"), ("E", "Impossible")],
    },
    {
        "id": 3,
        "text": "Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
    {
        "id": 4,
        "text": "Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
    {
        "id": 5,
        "text": "Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
    {
        "id": 6,
        "text": "Does your child follow where you’re looking?",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
    {
        "id": 7,
        "text": "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
        "options": [("A", "Always"), ("B", "Usually"), ("C", "Sometimes"), ("D", "Rarely"), ("E", "Never")],
    },
    {
        "id": 8,
        "text": "Would you describe your child’s first words as:",
        "options": [("A", "Very typical"), ("B", "Quite typical"), ("C", "Slightly unusual"), ("D", "Very unusual"), ("E", "My child doesn’t speak")],
    },
    {
        "id": 9,
        "text": "Does your child use simple gestures? (e.g. wave goodbye)",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
    {
        "id": 10,
        "text": "Does your child stare at nothing with no apparent purpose?",
        "options": [("A", "Many times a day"), ("B", "A few times a day"), ("C", "A few times a week"), ("D", "Less than once a week"), ("E", "Never")],
    },
]

ETHNICITY_MAP = {
    "white-european": 0,
    "asian": 1,
    "middle eastern": 2,
    "black": 3,
    "south asian": 4,
    "hispanic": 5,
    "latino": 6,
    "others": 7,
    "mixed": 8,
    "pacifica": 9,
    "native indian": 10,
}

def score_qchat(selected_options):
    scores = []
    for idx, selected in enumerate(selected_options, start=1):
        selected = (selected or "").upper()
        if idx <= 9:
            scores.append(1 if selected in {"C", "D", "E"} else 0)
        else:
            scores.append(1 if selected in {"A", "B", "C"} else 0)
    return scores

def build_model_features(form):
    selected_options = [form.get(f"A{i}", "") for i in range(1, 11)]
    scored_answers = score_qchat(selected_options)
    child_sex = 1 if form.get("child_sex") == "male" else 0
    ethnicity = ETHNICITY_MAP.get(form.get("child_ethnicity", "").lower(), 7)
    jaundice = 1 if form.get("jaundice") == "yes" else 0
    family_asd = 1 if form.get("family_asd") == "yes" else 0
    features = scored_answers + [child_sex, ethnicity, jaundice, family_asd]
    return selected_options, scored_answers, features

def predict_with_model(features):
    if QCHAT_MODEL is None:
        return None, None
    pred = int(QCHAT_MODEL.predict([features])[0])
    prob = None
    if hasattr(QCHAT_MODEL, "predict_proba"):
        prob = float(QCHAT_MODEL.predict_proba([features])[0][1])
    return pred, prob


@views.route("/")
def landing():
    return render_template("landing.html", title="Auto-Ism")


@views.route("/about")
def about():
    return render_template("about.html", title="About Us")


@views.route("/what-is-autism")
def what_is_autism():
    return render_template("what_is_autism.html", title="What's Autism")


@views.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.firstname = request.form.get("firstName", "").strip()
        current_user.lastname = request.form.get("lastName", "").strip()
        current_user.email = request.form.get("email", "").strip()
        current_user.dob = request.form.get("dob", "").strip()
        current_user.sex = request.form.get("sex", "").strip()
        db.session.commit()
        flash("Profile updated.", "success")
    return render_template("profile.html", title="Profile", user=current_user)


@views.route("/cases", methods=["GET", "POST"])
@login_required
def cases():
    if request.method == "POST":
        child_name = request.form.get("child_name", "").strip()
        child_dob = request.form.get("child_dob", "").strip()
        brief = request.form.get("brief", "").strip()

        if not child_name or not child_dob or not brief:
            flash("Please fill all case fields.", "error")
        else:
            new_case = Case(
                child_name=child_name,
                child_dob=child_dob,
                brief=brief,
                owner_id=current_user.id,
            )
            db.session.add(new_case)
            db.session.commit()
            flash("Case created successfully.", "success")

    user_cases = Case.query.filter_by(owner_id=current_user.id).order_by(Case.created_at.desc()).all()
    return render_template("cases/list.html", title="Cases", cases=user_cases)


@views.route("/cases/<int:case_id>/edit", methods=["GET", "POST"])
@login_required
def edit_case(case_id):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != current_user.id:
        abort(403)

    if request.method == "POST":
        child_name = request.form.get("child_name", "").strip()
        child_dob = request.form.get("child_dob", "").strip()
        brief = request.form.get("brief", "").strip()

        if not child_name or not child_dob or not brief:
            flash("Please fill all case fields.", "error")
        else:
            case.child_name = child_name
            case.child_dob = child_dob
            case.brief = brief
            db.session.commit()
            flash("Case updated successfully.", "success")
            return redirect(url_for("views.cases"))

    return render_template("cases/edit.html", title="Edit Case", case=case)


@views.route("/cases/<int:case_id>/delete", methods=["POST"])
@login_required
def delete_case(case_id):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != current_user.id:
        abort(403)

    db.session.delete(case)
    db.session.commit()
    flash("Case deleted.", "success")
    return redirect(url_for("views.cases"))


@views.route("/cases/<int:case_id>/test", methods=["GET", "POST"])
@login_required
def test_case(case_id):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != current_user.id:
        abort(403)

    spark_score = None
    prediction = None
    probability = None
    selected_answers = {}

    if request.method == "POST":
        selected_options, scored_answers, features = build_model_features(request.form)
        spark_score = sum(scored_answers)
        prediction, probability = predict_with_model(features)

        selected_answers = {f"A{i}": selected_options[i-1] for i in range(1, 11)}

        label = "Autism likelihood detected" if prediction == 1 else "Low autism likelihood"
        prob_text = f"{probability * 100:.1f}%" if probability is not None else "N/A"

        result = TestResult(
            case_id=case.id,
            spark_score=spark_score,
            image_score=None,
            combined_risk=probability,
            notes=f"{label} | Model probability: {prob_text}"
        )
        db.session.add(result)
        case.last_result_summary = f"Q-CHAT score {spark_score}/10 | Risk {prob_text}"
        db.session.commit()
        flash("Assessment submitted successfully.", "success")

    return render_template(
        "cases/test.html",
        title="Test Case",
        case=case,
        spark_score=spark_score,
        prediction=prediction,
        probability=probability,
        questions=QCHAT_QUESTIONS,
        selected_answers=selected_answers,
        ethnicities=list(ETHNICITY_MAP.keys()),
    )


@views.route("/cases/<int:case_id>/results")
@login_required
def case_results(case_id):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != current_user.id:
        abort(403)

    results = TestResult.query.filter_by(case_id=case.id).order_by(TestResult.created_at.desc()).all()
    return render_template("cases/results.html", title="Results", case=case, results=results)


@views.route("/cases/<int:case_id>/report")
@login_required
def case_report(case_id):
    case = Case.query.get_or_404(case_id)
    if case.owner_id != current_user.id:
        abort(403)

    latest_result = (
        TestResult.query.filter_by(case_id=case.id)
        .order_by(TestResult.created_at.desc())
        .first()
    )
    return render_template("cases/report.html", title="Detailed Report", case=case, result=latest_result)
