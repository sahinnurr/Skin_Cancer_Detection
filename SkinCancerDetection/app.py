from flask import Flask, render_template
from config import Config
from routes import auth_bp
from flask_jwt_extended import JWTManager
from models import db, create_db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    create_db(app)

    JWTManager(app)

    # Registering blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')

    @app.route('/')
    def home():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
