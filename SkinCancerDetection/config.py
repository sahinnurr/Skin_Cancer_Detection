import os

class Config:
    SECRET_KEY = os.urandom(24)
    JWT_SECRET_KEY = 'your_jwt_secret_key'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour

    # Database configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
