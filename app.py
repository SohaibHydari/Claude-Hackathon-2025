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

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1-mini")
RECIPE_MODEL = os.getenv("RECIPE_MODEL", "gpt-4.1-mini")


# --- Helpers ----------------------------------------------------------------

def _clean_json_text(raw: str) -> str:
    """
    Strip common wrappers like ```json ... ``` so json.loads() works.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def detect_ingredients(img_bytes):
    """
    Use a vision-capable model to list visible ingredients.
    Returns a cleaned list of ingredient names.
    """
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

    data = json.loads(raw)

    # Allow either a bare list or {"ingredients": [...]}
    if isinstance(data, dict) and "ingredients" in data:
        ingredients = data["ingredients"]
    else:
        ingredients = data

    if not isinstance(ingredients, list):
        raise ValueError("Vision model did not return a list of ingredients")

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
    Generate recipes that use mostly the existing ingredients.
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
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content
    raw = _clean_json_text(raw)

    data = json.loads(raw)
    recipes = data.get("recipes")
    if not isinstance(recipes, list):
        raise ValueError("Model did not return 'recipes' list in JSON")

    normalized = []
    for r in recipes:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "title": (r.get("title") or "").strip() or "Untitled Recipe",
                "short_description": (r.get("short_description") or "").strip(),
                "ingredients_used": r.get("ingredients_used", []),
                "steps": r.get("steps", []),
            }
        )

    return normalized


def generate_stretch_recipes(ingredients):
    """
    Suggest 'almost possible' recipes that require picking up a few extra items
    from the store.

    Returns a list of dicts:
      - title
      - short_description
      - ingredients_used_from_fridge
      - extra_ingredients_to_buy
      - steps
    """
    ingredients_str = ", ".join(ingredients) if ingredients else "nothing obvious"

    prompt = f"""
You are a home cooking coach.

The user currently has these ingredients in their fridge:
{ingredients_str}

They are willing to make a quick trip to the store to buy a FEW extra items.

Tasks:
1. Suggest 3 appealing recipes that are ALMOST possible with what they have now,
   but require up to 3 additional ingredients each.
2. For each recipe, include:
   - title
   - short_description
   - ingredients_used_from_fridge: which of the existing ingredients are used
   - extra_ingredients_to_buy: up to 3 additional ingredients they should buy
   - steps: 4–6 concise numbered steps

Be realistic and try to reuse as many existing ingredients as possible.

Respond ONLY in strict JSON with this shape:

{{
  "stretch_recipes": [
    {{
      "title": "string",
      "short_description": "string",
      "ingredients_used_from_fridge": ["string", "..."],
      "extra_ingredients_to_buy": ["string", "..."],
      "steps": ["string", "..."]
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=RECIPE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content
    raw = _clean_json_text(raw)

    data = json.loads(raw)
    stretch = data.get("stretch_recipes")
    if not isinstance(stretch, list):
        raise ValueError("Model did not return 'stretch_recipes' list in JSON")

    normalized = []
    for r in stretch:
        if not isinstance(r, dict):
            continue
        normalized.append(
            {
                "title": (r.get("title") or "").strip() or "Upgraded Recipe Idea",
                "short_description": (r.get("short_description") or "").strip(),
                "ingredients_used_from_fridge": r.get("ingredients_used_from_fridge", []),
                "extra_ingredients_to_buy": r.get("extra_ingredients_to_buy", []),
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
        ingredients = detect_ingredients(img_bytes)
        recipes = generate_recipes(ingredients)
        stretch_recipes = generate_stretch_recipes(ingredients)

        return jsonify(
            {
                "ingredients": ingredients,
                "recipes": recipes,
                "stretch_recipes": stretch_recipes,
            }
        )

    except json.JSONDecodeError as e:
        print("JSON parse error from model:", e)
        return (
            jsonify(
                {"error": "AI response could not be parsed as JSON. Try again in a moment."}
            ),
            500,
        )
    except Exception as e:
        print("Error in /analyze:", repr(e))
        return jsonify({"error": "Something went wrong while analyzing the image."}), 500


# --- Entrypoint -------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
