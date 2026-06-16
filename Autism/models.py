from . import db
from flask_login import UserMixin
from datetime import datetime


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    firstname = db.Column(db.String(150), nullable=False)
    lastname = db.Column(db.String(150), nullable=False)
    dob = db.Column(db.String(50))
    sex = db.Column(db.String(50))
    profile_image = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    cases = db.relationship('Case', backref='owner', lazy=True, cascade='all, delete-orphan')


class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    child_name = db.Column(db.String(150), nullable=False)
    child_dob = db.Column(db.String(50), nullable=False)
    brief = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_result_summary = db.Column(db.String(200))
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    tests = db.relationship('TestResult', backref='case', lazy=True, cascade='all, delete-orphan')


class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey('case.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    spark_score = db.Column(db.Integer)
    image_score = db.Column(db.Float)
    combined_risk = db.Column(db.Float)
    prediction_label = db.Column(db.String(120))
    notes = db.Column(db.String(1000))
    answers_json = db.Column(db.Text)
    report_text = db.Column(db.Text)


class GameScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.Integer, db.ForeignKey('case.id'), nullable=False)
    game = db.Column(db.String(50), nullable=False)
    level = db.Column(db.String(20))
    score = db.Column(db.Integer, nullable=False)
    max_score = db.Column(db.Integer, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    case = db.relationship('Case', backref=db.backref('game_scores', lazy=True, cascade='all, delete-orphan'))

    __table_args__ = (db.UniqueConstraint('case_id', 'game', name='uq_case_game'),)
