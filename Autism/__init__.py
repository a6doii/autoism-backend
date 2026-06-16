from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash
from os import path, makedirs


db = SQLAlchemy()
BASE_DIR = path.dirname(__file__)
DB_PATH = path.join(BASE_DIR, 'database.db')


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'auto-ism-react-flask-secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JSON_SORT_KEYS'] = False

    upload_dir = path.join(BASE_DIR, 'static', 'uploads')
    makedirs(upload_dir, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = upload_dir

    db.init_app(app)
    CORS(app, supports_credentials=True, origins=[
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'https://autoism-backend-production.up.railway.app',
        'https://autoism-rdb9qtwo1-a6doiis-projects.vercel.app',
    ])

    @app.after_request
    def add_no_cache_headers(response):
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    from .auth import auth
    from .api import api
    app.register_blueprint(auth, url_prefix='/api')
    app.register_blueprint(api, url_prefix='/api')

    from .models import User

    with app.app_context():
        db.create_all()

        existing_columns = [col['name'] for col in inspect(db.engine).get_columns('user')]
        if 'profile_image' not in existing_columns:
            with db.engine.begin() as conn:
                conn.execute(text('ALTER TABLE user ADD COLUMN profile_image VARCHAR(255)'))

        admin = User.query.filter_by(email='admin1').first()
        if not admin:
            admin = User(
                email='admin1',
                firstname='Owner',
                lastname='Dashboard',
                dob='',
                sex='',
                is_admin=True,
                password=generate_password_hash('s7s1234567A', method='pbkdf2:sha256', salt_length=16),
            )
            db.session.add(admin)
            db.session.commit()

    login_manager = LoginManager()
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        from .models import User
        return User.query.get(int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        return jsonify({'error': 'Authentication required'}), 401

    return app
