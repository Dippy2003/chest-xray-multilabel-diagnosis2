const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const resultDiv = document.getElementById("result");
const statusEl = document.getElementById("status");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  statusEl.textContent = "Running inference...";
  resultDiv.classList.add("hidden");

  document.getElementById("preview-img").src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    renderResult(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});

function renderResult(data) {
  const tbody = document.querySelector("#predictions-table tbody");
  tbody.innerHTML = "";

  for (const [className, info] of Object.entries(data.predictions)) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${className}</td>
      <td>${info.probability.toFixed(4)}</td>
      <td>${info.predicted ? "yes" : "no"}</td>
    `;
    tbody.appendChild(row);
  }

  document.getElementById("gradcam-class").textContent = data.gradcam.class;
  document.getElementById("gradcam-img").src = `data:image/png;base64,${data.gradcam.image_base64}`;

  resultDiv.classList.remove("hidden");
}
