# app.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file into environment variables
load_dotenv()

app = Flask(__name__)

# Read from environment (NOT from code)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Did you create a .env file?")

client = OpenAI(api_key=OPENAI_API_KEY)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_bytes = img_file.read()

    # 1) Call vision model to detect ingredients
    ingredients = detect_ingredients(img_bytes)

    # 2) Call LLM to generate recipes
    recipes = generate_recipes(ingredients)

    return jsonify({
        "ingredients": ingredients,
        "recipes": recipes
    })


def detect_ingredients(img_bytes):
    # ---- Vision model call ----
    # Pseudocode: adapt to your provider’s API
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # must support images if you're actually doing vision here
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "You are a kitchen assistant. "
                        "Look carefully at this fridge photo and list visible, usable food ingredients. "
                        "Return ONLY a JSON array of short ingredient names, e.g. "
                        '["milk", "eggs", "spinach"].'
                    )},
                    {"type": "image", "image": img_bytes}  # exact field depends on the SDK
                ]
            }
        ]
    )
    raw = response.choices[0].message.content
    """
    # For hackathon demo (no time to parse JSON safely), you can even ask for plain text and split.
    # But trying JSON is nicer; just catch errors.

    # MOCK for now (so you can dev the rest without the real call)
    import json
    raw = '["milk", "eggs", "cheddar cheese", "spinach"]'

    try:
        ingredients = json.loads(raw)
    except Exception:
        # naive fallback: split by commas
        ingredients = [x.strip() for x in raw.split(",")]

    # Clean a bit
    unique = []
    for ing in ingredients:
        ing_clean = ing.lower().strip()
        if ing_clean and ing_clean not in unique:
            unique.append(ing_clean)

    return unique


def generate_recipes(ingredients):
    # ---- LLM call ----
    ingredients_str = ", ".join(ingredients)

    prompt = f"""
    You are an inventive home cook assistant.

    User has the following ingredients in their fridge:
    {ingredients_str}

    1. Suggest 3 simple recipes they can cook using mostly these ingredients.
    2. For each recipe, return:
       - title
       - short_description
       - ingredients_used (subset of the given ingredients)
       - steps: 4–6 concise steps

    Respond in strict JSON with this shape:
    {{
      "recipes": [
        {{
          "title": "...",
          "short_description": "...",
          "ingredients_used": ["...", "..."],
          "steps": ["...", "..."]
        }}
      ]
    }}
    """

    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    """
    # MOCK for dev:
    raw = """
    {
      "recipes": [
        {
          "title": "Cheesy Spinach Omelette",
          "short_description": "A quick breakfast omelette stuffed with spinach and cheddar.",
          "ingredients_used": ["eggs", "cheddar cheese", "spinach"],
          "steps": [
            "Beat the eggs in a bowl.",
            "Pour into a hot pan and cook until just set.",
            "Add chopped spinach and grated cheddar on one half.",
            "Fold, cook another minute, then serve."
          ]
        }
      ]
    }
    """

    import json
    try:
        data = json.loads(raw)
        return data.get("recipes", [])
    except Exception:
        # Fallback: return a single very simple recipe if parsing fails
        return [{
            "title": "Freestyle Fridge Scramble",
            "short_description": "Throw everything in a pan and make a hearty scramble.",
            "ingredients_used": ingredients,
            "steps": [
                "Chop all ingredients into bite-sized pieces.",
                "Heat some oil or butter in a pan.",
                "Add everything and cook until heated through.",
                "Season to taste and serve."
            ]
        }]


if __name__ == "__main__":
    app.run(debug=True)
