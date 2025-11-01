from datetime import datetime
from flask import Blueprint, request, jsonify, Flask
from .main import generate_bot_reply 

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/chat/<id>", methods=["POST"])
def chat(id):
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    bot_response = generate_bot_reply (user_message, id)
    return jsonify({"response": bot_response, "session_id":id})

@chatbot_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "uptime": "Serviço ativo"
    })