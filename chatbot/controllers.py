from .services import generate_bot_reply

def save_log(user_message: str, bot_reply: str):
    with open("chat_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"Usuário: {user_message}\nBot: {bot_reply}\n---\n")

def process_message(user_message: str) -> str:
    bot_reply = generate_bot_reply(user_message)
    save_log(user_message, bot_reply)
    return bot_reply