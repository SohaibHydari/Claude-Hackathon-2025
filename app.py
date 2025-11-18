# app.py
from flask import Flask, render_template, request, jsonify
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load .env

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in your .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# ====== HOME PAGE ======
@app.route("/")
def index():
    return render_template("index.html")


# ====== MAIN ANALYSIS ENDPOINT ======
@app.route("/analyze", methods=["POST"])
def analyze():
    # Validate image upload
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_bytes = img_file.read()

    # Step 1 — Detect ingredients via vision
    ingredients = detect_ingredients(img_bytes)

    # Step 2 — Generate recipes
    recipes = generate_recipes(ingredients)

    return jsonify({
        "ingredients": ingredients,
        "recipes": recipes
    })


# ---------------------------------------------------------
# 🧠 IMAGE → INGREDIENTS
# ---------------------------------------------------------
def detect_ingredients(img_bytes):
    MOCK_VISION = False  # set True if testing without API

    if MOCK_VISION:
        raw = '["milk", "eggs", "spinach", "cheddar cheese"]'
    else:
        # --- Actual Vision API Call (Correct Format!) ---
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Look at this fridge photo and return ONLY a JSON array "
                                "of visible food ingredients. Example: "
                                '["milk", "eggs", "cheddar"].'
                            )
                        },
                        {
                            "type": "input_image",
                            "image": img_bytes
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content

    # Parse JSON safely
    try:
        ingredients = json.loads(raw)
    except:
        # Fallback: rough split
        ingredients = [x.strip() for x in raw.split(",")]

    # Clean + dedupe
    cleaned = []
    for i in ingredients:
        x = i.lower().strip()
        if x and x not in cleaned:
            cleaned.append(x)

    return cleaned


# ---------------------------------------------------------
# 🍳 INGREDIENTS → RECIPES
# ---------------------------------------------------------
def generate_recipes(ingredients):
    MOCK_LLM = False  # set to True for local testing

    ingredients_str = ", ".join(ingredients)

    if MOCK_LLM:
        raw = """
        {
          "recipes": [
            {
              "title": "Cheesy Spinach Omelette",
              "short_description": "A quick breakfast omelette with spinach and cheddar.",
              "ingredients_used": ["eggs", "spinach", "cheddar cheese"],
              "steps": [
                "Beat the eggs.",
                "Cook in pan until soft.",
                "Add spinach + cheese.",
                "Fold and serve."
              ]
            }
          ]
        }
        """
    else:
        prompt = f"""
        You are a helpful cooking assistant.

        The user has these ingredients:
        {ingredients_str}

        Create exactly 3 simple recipes using mostly these ingredients.
        Return STRICT JSON in this shape:

        {{
          "recipes": [
            {{
              "title": "...",
              "short_description": "...",
              "ingredients_used": ["..."],
              "steps": ["...", "..."]
            }}
          ]
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
        return data.get("recipes", [])
    except:
        # Last-resort fallback
        return [{
            "title": "Freestyle Scramble",
            "short_description": "A simple dish using whatever is available.",
            "ingredients_used": ingredients,
            "steps": [
                "Chop everything.",
                "Heat oil in a pan.",
                "Add ingredients and cook.",
                "Season and serve."
            ]
        }]


if __name__ == "__main__":
    app.run(debug=True)
