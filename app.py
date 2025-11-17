from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_bytes = img_file.read()

    ingredients = detect_ingredients(img_bytes)
    recipes = generate_recipes(ingredients)

    return jsonify({"ingredients": ingredients, "recipes": recipes})


# Mock ingredient detection
def detect_ingredients(img_bytes):
    return ["eggs", "milk", "spinach"]


# Mock recipe generation
def generate_recipes(ingredients):
    return [
        {
            "title": "Spinach Omelette",
            "short_description": "Fast, fluffy, delicious.",
            "ingredients_used": ["eggs", "spinach"],
            "steps": [
                "Whisk eggs.",
                "Cook in pan.",
                "Add spinach.",
                "Fold and serve."
            ]
        }
    ]


if __name__ == "__main__":
    app.run(debug=True)
