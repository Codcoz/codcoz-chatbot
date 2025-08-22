from flask import Blueprint, request, jsonify
from .controllers import process_message

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Mensagem vazia"}), 400

    bot_response = process_message(user_message)
    return jsonify({"response": bot_response})