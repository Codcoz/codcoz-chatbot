from flask import Blueprint, request, jsonify
from .main import process_message

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/chat/<id>", methods=["POST"])
def chat(id):
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    bot_response = process_message(user_message, id)
    return jsonify({"response": bot_response, "session_id":id})