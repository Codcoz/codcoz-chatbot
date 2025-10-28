from flask import Flask
from chatbot.routes import chatbot_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(chatbot_bp)  # expõe /chat e /health
    return app

# exporta a instância para o Flask CLI quando FLASK_APP=app.py
app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)