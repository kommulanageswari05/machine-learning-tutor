import os
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import google.generativeai as genai


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# ============================================================
# LOAD ENV
# ============================================================

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )
else:
    client = None


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder=None
)


# ============================================================
# GEMINI HELPER
# ============================================================

def ask_gemini(prompt):

    if client is None:
        raise RuntimeError(
            "Gemini API is not configured. "
            "Please add GEMINI_API_KEY to .env"
        )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response is None:
            raise RuntimeError(
                "Gemini returned no response."
            )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()

    except Exception as e:

        print("Gemini Error:", str(e))

        raise


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:].strip()

    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "running",
        "model": MODEL_NAME
    })


# ============================================================
# CONCEPT EXPLANATION
# ============================================================

@app.route("/api/explain", methods=["POST"])
def explain():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        topic = str(
            data.get("topic", "")
        ).strip()

        if not topic:

            return jsonify({
                "success": False,
                "error": "Please enter a topic."
            }), 400


        prompt = f"""
You are an expert Machine Learning teacher.

Explain this concept to a college student:

TOPIC:
{topic}

Use the following structure:

1. Simple Definition
2. How It Works
3. Important Concepts
4. Real World Applications
5. Advantages
6. Disadvantages
7. Simple Example
8. Exam Points

Rules:

- Use simple English.
- Assume the student is a beginner.
- Use clear headings.
- Use bullet points where useful.
- Stay focused on the requested topic.
- Do not explain unrelated concepts.
"""

        answer = ask_gemini(prompt)


        return jsonify({

            "success": True,

            "topic": topic,

            "explanation": answer

        })


    except Exception as e:

        print(
            "Explain Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# CODE GENERATION
# ============================================================

@app.route("/api/generate-code", methods=["POST"])
def generate_code():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        user_prompt = str(
            data.get("prompt", "")
        ).strip()

        if not user_prompt:

            return jsonify({
                "success": False,
                "error": "Please describe the code you want."
            }), 400


        prompt = f"""
You are an expert Python and Machine Learning developer.

Generate clean, beginner-friendly and executable Python code.

USER REQUEST:
{user_prompt}

Requirements:

- Use Python.
- Prefer scikit-learn for machine learning.
- Include necessary imports.
- Include sample data if appropriate.
- Add useful comments.
- Make the code executable.
- Explain important parts after the code.
"""

        answer = ask_gemini(prompt)


        return jsonify({

            "success": True,

            "code": answer

        })


    except Exception as e:

        print(
            "Code Generation Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# QUIZ GENERATION
# ============================================================

@app.route("/api/quiz", methods=["POST"])
def quiz():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        topic = str(
            data.get("topic", "")
        ).strip()

        if not topic:

            return jsonify({

                "success": False,

                "error": "Please enter a topic."

            }), 400


        prompt = f"""
Create exactly 5 multiple-choice questions about:

{topic}

Each question must contain:

- question
- options
- answer

Rules:

- Exactly 4 options.
- answer must be a number from 0 to 3.
- answer represents the correct option.
- Questions must be related to the topic.
- Questions should be suitable for college students.

Return ONLY valid JSON.

Example:

[
  {{
    "question": "What is Python?",
    "options": [
      "Programming language",
      "Database",
      "Operating system",
      "Browser"
    ],
    "answer": 0
  }}
]
"""

        answer = ask_gemini(prompt)

        cleaned = clean_json(answer)

        questions = json.loads(cleaned)


        if not isinstance(questions, list):

            raise ValueError(
                "Quiz response is not a list."
            )


        if len(questions) != 5:

            raise ValueError(
                "Gemini did not return exactly 5 questions."
            )


        return jsonify({

            "success": True,

            "topic": topic,

            "questions": questions

        })


    except json.JSONDecodeError:

        return jsonify({

            "success": False,

            "error": "Gemini returned invalid quiz JSON."

        }), 500


    except Exception as e:

        print(
            "Quiz Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# AI VOICE EXPLANATION
# ============================================================

@app.route("/api/voice-explanation", methods=["POST"])
def voice_explanation():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        topic = str(
            data.get("topic", "")
        ).strip()


        if not topic:

            return jsonify({

                "success": False,

                "error": "Please enter a topic."

            }), 400


        prompt = f"""
You are a friendly Machine Learning teacher.

The student wants to LISTEN to an explanation.

TOPIC:
{topic}

Create a natural spoken explanation for this exact topic.

Requirements:

- Explain only the requested topic.
- Start with a simple definition.
- Explain how it works step by step.
- Explain important concepts.
- Give one easy real-world example.
- Explain why it is useful.
- Mention important points to remember.
- Assume the student is a beginner.
- Use simple English.
- Make it sound like a teacher speaking to a student.
- Use natural conversational sentences.
- Do not use markdown.
- Do not use headings.
- Do not use bullet points.
- Do not use tables.
- Do not use emojis.
- Do not use code blocks.
- Do not mention these instructions.
- Keep it around 1 to 2 minutes.

Return ONLY the spoken explanation.
"""

        answer = ask_gemini(prompt)


        return jsonify({

            "success": True,

            "topic": topic,

            "explanation": answer

        })


    except Exception as e:

        print(
            "Voice Explanation Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# DYNAMIC VISUAL DIAGRAM
# ============================================================

@app.route("/api/diagram", methods=["POST"])
def diagram():

    try:

        data = request.get_json(
            silent=True
        ) or {}


        topic = str(
            data.get("topic", "")
        ).strip()


        if not topic:

            return jsonify({

                "success": False,

                "error": "Please enter a topic."

            }), 400


        prompt = f"""
You are an expert Machine Learning teacher.

Create a dynamic educational visual workflow for this topic:

TOPIC:
{topic}

The diagram MUST be specifically related to the requested topic.

Do NOT use a generic Machine Learning workflow.

For example:

If topic is Linear Regression:
Input Data → Features → Linear Model → Training → Prediction

If topic is CNN:
Image → Convolution → ReLU → Pooling → Flatten → Classification

If topic is K-Means:
Data → Choose K → Initialize Centroids → Assign Points → Update Centroids → Repeat → Clusters

If topic is Decision Tree:
Dataset → Feature Selection → Split → Branches → Leaf Nodes → Prediction

Return ONLY valid JSON.

Use exactly this structure:

{{
    "title": "Topic specific title",
    "description": "Short explanation of the workflow",
    "steps": [
        {{
            "title": "Step title",
            "icon": "emoji",
            "description": "Short simple explanation"
        }}
    ]
}}

Rules:

- Generate 4 to 8 steps.
- Steps must be specific to the requested topic.
- Steps must represent the actual learning/process flow.
- Do not always use Data, Preprocessing, Training, Prediction.
- Use a suitable emoji for every step.
- Keep descriptions short.
- Use simple English.
- Make the diagram understandable to a beginner.
- Return ONLY JSON.
"""

        answer = ask_gemini(prompt)

        cleaned = clean_json(answer)

        diagram_data = json.loads(cleaned)


        if not isinstance(diagram_data, dict):

            raise ValueError(
                "Diagram response is not an object."
            )


        steps = diagram_data.get(
            "steps",
            []
        )


        if not isinstance(steps, list):

            raise ValueError(
                "Diagram steps are invalid."
            )


        return jsonify({

            "success": True,

            "topic": topic,

            "diagram": diagram_data

        })


    except json.JSONDecodeError:

        return jsonify({

            "success": False,

            "error": "Gemini returned invalid diagram JSON."

        }), 500


    except Exception as e:

        print(
            "Diagram Error:",
            str(e)
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 404
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success": False,

        "error": "Route not found."

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("                  ML LearnAI")
    print("=" * 60)
    print("Server: http://127.0.0.1:5000")
    print("Model :", MODEL_NAME)
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
