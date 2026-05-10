// ==========================================
// 1. MENU RESPONSIVE & SCROLL
// ==========================================
var menu = document.querySelector(".menu");
var menu_toggle = document.querySelector(".menu_toggle");
menu_toggle.onclick = function () {
  menu_toggle.classList.toggle("active");
  menu.classList.toggle("responsive");
};

const homeLink = document.querySelector(".home_link");
if (homeLink) {
  homeLink.addEventListener("click", (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

// ==========================================
// 2. CONSOLE LOGIQUE (Affichage des messages)
// ==========================================
const outputConsole = document.getElementById("outputConsole");

function addLog(message, type = "info") {
  const log = document.createElement("p");
  log.classList.add("log", type);
  log.textContent = "> " + message;
  outputConsole.appendChild(log);
  // Scroll auto vers le bas
  outputConsole.scrollTop = outputConsole.scrollHeight;
}

document.getElementById("clearConsole").addEventListener("click", () => {
  outputConsole.innerHTML = '<p class="log info">> System ready...</p>';
});

// ==========================================
// 3. SÉLECTION DU MODÈLE
// ==========================================
const modelCards = document.querySelectorAll(".model_card");
let selectedModel = null;

modelCards.forEach((card) => {
  card.addEventListener("click", () => {
    modelCards.forEach((c) => c.classList.remove("active"));
    card.classList.add("active");
    selectedModel = parseInt(card.dataset.model);
    addLog(`Model ${selectedModel} selected.`, "info");
  });
});

// ==========================================
// 4. GESTION DES MODES D'ENTRÉE (Write vs File)
// ==========================================
const writeModeBtn = document.getElementById("writeMode");
const fileModeBtn = document.getElementById("fileMode");
const codeEditor = document.getElementById("codeEditor");
const fileInput = document.getElementById("fileInput");

// Mode "Write Code"
writeModeBtn.addEventListener("click", () => {
  writeModeBtn.classList.add("active");
  fileModeBtn.classList.remove("active");
  codeEditor.readOnly = false;
  codeEditor.placeholder = "Write your code here...";
  codeEditor.focus();
});

// Mode "Upload File"
fileModeBtn.addEventListener("click", () => {
  fileModeBtn.classList.add("active");
  writeModeBtn.classList.remove("active");
  // Simule un clic sur l'input type="file" caché
  fileInput.click();
});

// Lecture du fichier quand l'utilisateur le sélectionne
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  // Quand la lecture réussit
  reader.onload = function (e) {
    codeEditor.value = e.target.result;
    addLog(`File '${file.name}' loaded successfully.`, "info");
  };

  // En cas d'erreur de lecture
  reader.onerror = function () {
    addLog(`Error reading file '${file.name}'.`, "error");
  };

  // Lit le fichier comme du texte simple
  reader.readAsText(file);

  // Réinitialise l'input pour pouvoir re-sélectionner le même fichier si besoin
  event.target.value = "";
});

// Bouton Clear Code
document.getElementById("clearCode").addEventListener("click", () => {
  codeEditor.value = "";
  addLog("Code editor cleared.", "info");
});

// ==========================================
// 5. CONNEXION AU BACKEND (Bouton Run / Debug)
// ==========================================
document.getElementById("runCode").addEventListener("click", async () => {
  if (!selectedModel) {
    addLog("Error: Please select a model first!", "error");
    return;
  }

  const code = codeEditor.value.trim();

  if (code === "") {
    addLog("Warning: No code provided in the editor!", "warning");
    return;
  }

  addLog(`Running Model ${selectedModel}...`, "info");
  addLog("Extracting features with CodeBERT...", "info");

  // On désactive le bouton pendant le chargement pour éviter les double-clics
  const runBtn = document.getElementById("runCode");
  runBtn.disabled = true;
  runBtn.textContent = "Processing...";
  codeEditor.disabled = true;

  try {
    // Requête HTTP POST vers votre serveur Flask (app.py)
    console.log("🚀 Sending request to backend...");
    const response = await fetch("http://127.0.0.1:8001/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_id: parseInt(selectedModel),
        code: code,
      }),
    });
    console.log("✅ Response received");

    const data = await response.json();

    if (response.ok) {
      addLog("Quantum Inference complete 🚀", "info");
      // Affichage du résultat renvoyé par le modèle Python
      // Les types (error/success) gèreront la couleur si vos CSS sont bien configurés
      addLog(
        `Result: ${data.message} | Confidence: ${data.confidence ?? "?"}`,
        data.type,
      );
    } else {
      addLog(`Backend Error: ${data.error}`, "error");
    }
  } catch (error) {
    addLog(
      "Connection to AI Server failed! Make sure the Python backend (app.py) is running.",
      "error",
    );
    console.error("Fetch error:", error);
  } finally {
    // On réactive les contrôles
    runBtn.disabled = false;
    runBtn.textContent = "Run / Debug";
    codeEditor.disabled = false;
  }
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10000);
});

window.onload = function () {
  const code = localStorage.getItem("rerun_code");
  const model = localStorage.getItem("rerun_model");

  if (code) {
    document.getElementById("codeInput").value = code;
    document.getElementById("modelSelect").value = model;

    // nettoyer après usage
    localStorage.removeItem("rerun_code");
    localStorage.removeItem("rerun_model");
  }
};
