import os

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)

ALLOWED_ORIGIN = os.environ.get(
    "ALLOWED_ORIGIN",
    "https://manishchavan55.github.io",
)

CORS(
    app,
    resources={
        r"/chat": {
            "origins": [ALLOWED_ORIGIN],
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)

JARVIS_SYSTEM_PROMPT = """
You are Saurabh's portfolio AI assistant.

Your job is to answer questions about Saurabh Chavan's public portfolio,
projects, technical skills, education, experience, and how someone can
contact or hire him. You may also answer general technical questions when
helpful, especially Java, Spring Boot, REST APIs, SQL, JavaScript, React,
HTML, CSS, Git, and software development.

PORTFOLIO FACTS:
- Saurabh Chavan is a software engineer at Tata Consultancy Services (TCS).
- His public technical profile is Java-focused and includes Core Java, OOP,
  Java 8, Collections, JDBC, Spring Boot, Hibernate/JPA, REST APIs, SQL/MySQL,
  ReactJS, JavaScript, HTML, CSS, Maven, Git and GitHub.
- Public portfolio projects include a Bank Management System, Employee
  Management System, and Personal Portfolio.
- He has a B.Sc. in Computer Science from Deogiri College.
- The portfolio intentionally does not expose confidential client/project
  details from professional work.

RULES:
- Answer directly and naturally.
- Do not invent private work details, employers, client names, project metrics,
  certifications, or achievements that are not in the public portfolio data.
- If asked about confidential TCS work, explain that the portfolio does not
  disclose confidential client or project details.
- If asked how to hire/contact Saurabh, point the user to the portfolio's
  contact or Hire Me section.
- For technical questions, give practical explanations and examples.
- If the user wants to learn something, teach it step by step.
- If the user asks a follow-up such as "why?" or "continue", use the current
  conversation context when it is available.
- Keep normal answers concise, but provide enough detail to be useful.
""".strip()

OPENROUTER_URL = os.environ.get(
    "AI_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)
MODEL = os.environ.get("AI_MODEL", "openrouter/free")


def get_api_key():
    return os.environ.get("AI_API_KEY", "").strip()


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "Saurabh Portfolio AI API",
        "model": MODEL,
    }), 200


@app.get("/health")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "Saurabh Portfolio AI API",
        "model": MODEL,
    }), 200


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    api_key = get_api_key()
    if not api_key:
        app.logger.error("AI_API_KEY is not configured on the server.")
        return jsonify({"error": "AI service is not configured."}), 500

    messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]

    if isinstance(history, list):
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                content = content.strip()
                if content:
                    messages.append({"role": role, "content": content[:4000]})

    messages.append({"role": "user", "content": message[:4000]})

    try:
        response = requests.post(
            OPENROUTER_URL,
            json={"model": MODEL, "messages": messages},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://manishchavan55.github.io/Portfolio-Saurabh/",
                "X-Title": "Saurabh Chavan Portfolio AI",
            },
            timeout=60,
        )

        if not response.ok:
            app.logger.error(
                "OpenRouter error %s: %s",
                response.status_code,
                response.text,
            )
            return jsonify({
                "error": f"AI provider returned HTTP {response.status_code}."
            }), 502

        result = response.json()
        reply = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not isinstance(reply, str) or not reply.strip():
            app.logger.error("OpenRouter returned no text: %s", result)
            return jsonify({"error": "AI returned an empty response."}), 502

        return jsonify({"response": reply.strip()}), 200

    except requests.exceptions.Timeout:
        app.logger.exception("OpenRouter request timed out.")
        return jsonify({"error": "The AI service took too long to respond."}), 504

    except requests.exceptions.RequestException:
        app.logger.exception("OpenRouter connection failed.")
        return jsonify({"error": "Failed to communicate with the AI service."}), 502

    except ValueError:
        app.logger.exception("OpenRouter returned invalid JSON.")
        return jsonify({"error": "The AI service returned invalid data."}), 502

    except Exception:
        app.logger.exception("Unexpected server error.")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
