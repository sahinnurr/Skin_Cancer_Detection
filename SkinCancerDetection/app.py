from flask import Flask, render_template
from config import Config
from models import db
from routes import auth_bp
from flask_jwt_extended import JWTManager

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    JWTManager(app)

    with app.app_context():
        db.create_all()

    # Registering blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')

    @app.route('/')
    def home():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

