from flask import Flask, render_template, request, jsonify
import os
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env (for local dev)
load_dotenv()

app = Flask(__name__)

# --- Configuration & client -------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Did you create a .env file?")

client = OpenAI(api_key=OPENAI_API_KEY)

# You can override these via env vars if you want
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1-mini")
RECIPE_MODEL = os.getenv("RECIPE_MODEL", "gpt-4.1-mini")


# --- Helpers ----------------------------------------------------------------

def _clean_json_text(raw: str) -> str:
    """
    Strip common wrappers like ```json ... ``` so json.loads() doesn't explode.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        # Drop opening fence
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Drop closing fence
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def detect_ingredients(img_bytes):
    """
    Call an OpenAI vision-capable model to list visible ingredients in the fridge.
    Returns a cleaned list of ingredient names (lowercase, deduped).
    """
    # Encode the image as base64 and send as a data URL
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = (
        "You are a kitchen assistant. Look carefully at this fridge photo and "
        "list the visible, usable food ingredients (produce, dairy, drinks, "
        "packaged foods, etc.). Ignore non-food items and brand names unless "
        "they clearly indicate the food type (e.g., tofu, yogurt).\n\n"
        "Return ONLY a JSON array of short, generic ingredient names, for example:\n"
        '["eggs", "spinach", "yogurt", "tofu"]'
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                ],
            }
        ],
    )

    raw = response.choices[0].message.content
    raw = _clean_json_text(raw)

    data = json.loads(raw)  # let this raise if bad JSON – front-end will show error

    # Normalize to a simple list of strings
    if isinstance(data, dict) and "ingredients" in data:
        ingredients = data["ingredients"]
    else:
        ingredients = data

    if not isinstance(ingredients, list):
        raise ValueError("Vision model did not return a list of ingredients")

    # Clean & dedupe
    unique = []
    for ing in ingredients:
        if not isinstance(ing, str):
            continue
        ing_clean = ing.strip().lower()
        if ing_clean and ing_clean not in unique:
            unique.append(ing_clean)

    return unique


def generate_recipes(ingredients):
    """
    Call an OpenAI text model to generate recipes using the detected ingredients.
    Returns a list of recipe dicts with keys:
      - title
      - short_description
      - ingredients_used
      - steps
    """
    ingredients_str = ", ".join(ingredients) if ingredients else "nothing obvious"

    prompt = f"""
You are an inventive yet practical home cook assistant.

The user has the following ingredients in their fridge:
{ingredients_str}

Tasks:
1. Suggest 3 simple recipes they can reasonably cook using mostly these ingredients.
2. For each recipe, include:
   - title
   - short_description
   - ingredients_used (a subset of the given ingredients, plus basic pantry items like oil, salt, pepper if needed)
   - steps: 4–6 concise numbered steps

Respond ONLY in strict JSON with this shape:

{{
  "recipes": [
    {{
      "title": "string",
      "short_description": "string",
      "ingredients_used": ["string", "..."],
      "steps": ["string", "..."]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=RECIPE_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    raw = response.choices[0].message.content
    raw = _clean_json_text(raw)

    data = json.loads(raw)
    recipes = data.get("recipes")
    if not isinstance(recipes, list):
        raise ValueError("Model did not return 'recipes' list in JSON")

    # Optional: light normalization
    normalized = []
    for r in recipes:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "title": r.get("title", "").strip() or "Untitled Recipe",
                "short_description": r.get("short_description", "").strip(),
                "ingredients_used": r.get("ingredients_used", []),
                "steps": r.get("steps", []),
            }
        )

    return normalized


# --- Routes -----------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_bytes = img_file.read()

    if not img_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    try:
        # 1) Vision: detect ingredients
        ingredients = detect_ingredients(img_bytes)

        # 2) Text: generate recipes
        recipes = generate_recipes(ingredients)

        return jsonify(
            {
                "ingredients": ingredients,
                "recipes": recipes,
            }
        )

    except json.JSONDecodeError as e:
        # Specifically catch JSON parsing issues from the model
        print("JSON parse error from model:", e)
        return (
            jsonify(
                {
                    "error": "AI response could not be parsed as JSON. Try again in a moment."
                }
            ),
            500,
        )
    except Exception as e:
        # Generic catch-all so the frontend gets a clean error
        print("Error in /analyze:", repr(e))
        return jsonify({"error": "Something went wrong while analyzing the image."}), 500


# --- Entrypoint -------------------------------------------------------------

if __name__ == "__main__":
    # For local dev; in production, use a proper WSGI server
    app.run(debug=True)
