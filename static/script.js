const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const statusDiv = document.getElementById("status");
const ingredientsSection = document.getElementById("ingredients-section");
const ingredientsList = document.getElementById("ingredients-list");
const recipesSection = document.getElementById("recipes-section");
const recipesList = document.getElementById("recipes-list");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = imageInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);

  statusDiv.textContent = "Analyzing…";
  ingredientsSection.classList.add("hidden");
  recipesSection.classList.add("hidden");

  const res = await fetch("/analyze", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  const { ingredients, recipes } = data;

  // Render ingredients
  if (ingredients.length) {
    ingredientsSection.classList.remove("hidden");
    ingredientsList.innerHTML = ingredients
      .map(i => `<span class="pill">${i}</span>`)
      .join("");
  }

  // Render recipes
  if (recipes.length) {
    recipesSection.classList.remove("hidden");
    recipesList.innerHTML = recipes.map(renderRecipe).join("");
  }

  statusDiv.textContent = "";
});

function renderRecipe(recipe) {
  return `
    <div class="recipe-card">
      <h3>${recipe.title}</h3>
      <p>${recipe.short_description}</p>
      <h4>Ingredients used</h4>
      <ul>
        ${recipe.ingredients_used.map(i => `<li>${i}</li>`).join("")}
      </ul>
      <h4>Steps</h4>
      <ol>
        ${recipe.steps.map(s => `<li>${s}</li>`).join("")}
      </ol>
    </div>
  `;
}
