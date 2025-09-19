import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import docx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DOCX_PATH = os.getenv("DOCX_PATH", "flask_documentation.docx")

# Check for required environment variables at startup
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documentation_from_word(file_path):
    """Load and return documentation text from a Word (.docx) file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logger.error(f"Error loading documentation: {e}")
        return ""


app_documentation = load_documentation_from_word(DOCX_PATH)


def format_chat_history(history):
    """
    Convert chat history list of {"role": ..., "content": ...} to readable string.
    """
    if not isinstance(history, list):
        return ""
    formatted = []
    for msg in history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if role == "user":
            formatted.append(f"User: {content}")
        elif role in ("assistant", "bot"):
            formatted.append(f"AI: {content}")
    return "\n".join(formatted)


def error_response(message, status_code=400):
    return jsonify({"success": False, "error": message}), status_code


def success_response(answer, model, tokens):
    return jsonify({
        "success": True,
        "answer": answer,
        "model_used": model,
        "tokens_used": tokens
    })


@app.route('/ai-assistant', methods=['POST'])
def ai_assistant():
    try:
        data = request.get_json()
        user_question = data.get("question", "").strip()
        chat_history_raw = data.get("history", [])

        if not app_documentation:
            return error_response("Application documentation is missing or could not be loaded.", 500)
        if not user_question:
            return error_response("Missing 'question' in request.", 400)

        chat_history = format_chat_history(chat_history_raw)

        prompt = f"""
📘 You are a helpful and professional in-app assistant for a mobile or web application. Your task is to answer users' questions **only** using the official application documentation below.

---

### 📚 Application Documentation:
{app_documentation}

---

### 🧠 Conversation History:
{chat_history if chat_history else "No prior conversation."}

---

### ❓ User's Current Question:
{user_question}

---

### ✅ Instructions:
-**Answer in the same language the user used** (English or French).
- Answer using only the information in the documentation.
- Never mention the documentation or say "based on the docs"
- Never reveal you're an AI or bot
- Use a friendly, clear, and helpful tone, like a knowledgeable support agent.
- Avoid technical jargon unless it's present in the documentation.
- If the documentation contains a direct or inferred answer, respond concisely and professionally.
- If the answer is NOT in the documentation, respond exactly with:

    😊 "I can't help with that, but our support team can! 💬

- Do NOT guess, speculate, or invent information.
- Do NOT mention the documentation unless explicitly asked.
- Do NOT reveal you are an AI or language model; respond as a helpful assistant.
- Keep responses focused, informative, and approachable.
"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            # "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict assistant who answers only based on the provided documentation and returns helpful, professional responses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }

        response = requests.post(GROQ_URL, headers=headers, json=body)
        response.raise_for_status()

        result = response.json()
        message = result["choices"][0]["message"]["content"]

        return success_response(
            answer=message,
            model=result.get("model"),
            tokens=result.get("usage", {}).get("total_tokens", 0)
        )

    except Exception as e:
        logger.error(f"Error in /ai-assistant: {e}")
        return error_response(str(e), 500)


if __name__ == "__main__":
    app.run(debug=True)
