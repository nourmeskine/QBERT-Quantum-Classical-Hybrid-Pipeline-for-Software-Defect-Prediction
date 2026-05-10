const historyList = document.getElementById("historyList");
const searchInput = document.getElementById("searchInput");
const filterButtons = document.querySelectorAll(".filter_buttons button");

let historyData = [];
let currentFilter = "all";

// ================= FETCH =================
async function fetchHistory() {
  const res = await fetch("http://127.0.0.1:8001/history");
  historyData = await res.json();
  applyFilters();
}

// ================= RENDER =================
function renderHistory(data) {
  historyList.innerHTML = "";

  if (data.length === 0) {
    historyList.innerHTML = "<p>No history found</p>";
    return;
  }

  data.forEach((item, index) => {
    const card = document.createElement("div");
    card.classList.add("history_card");

    card.innerHTML = `
      <div class="card_top">
        <span class="model_tag">Model ${item.model}</span>
        <span>${item.result}</span>
      </div>

      <div class="code_preview">${item.code}</div>

      <div class="card_actions">
        <button class="rerun" onclick="rerun(${index})">Re-run</button>
        <button class="delete" onclick="deleteItem(${index})">Delete</button>
      </div>
    `;

    historyList.appendChild(card);
  });
}

// ================= DELETE =================
async function deleteItem(index) {
  await fetch(`http://127.0.0.1:8001/history/${index}`, {
    method: "DELETE",
  });

  fetchHistory();
}

// ================= RERUN =================
function rerun(index) {
  const item = historyData[index];

  // 🔥 redirection vers page debug avec data
  localStorage.setItem("rerun_code", item.code);
  localStorage.setItem("rerun_model", item.model);

  window.location.href = "index.html"; // page debug
}

// ================= SEARCH + FILTER =================
function applyFilters() {
  let filtered = [...historyData];

  // 🔍 SEARCH
  const searchValue = searchInput.value.toLowerCase();

  if (searchValue) {
    filtered = filtered.filter(
      (item) =>
        item.code.toLowerCase().includes(searchValue) ||
        item.result.toLowerCase().includes(searchValue) ||
        item.model.toString().includes(searchValue),
    );
  }

  // 🎯 FILTER MODEL
  if (currentFilter !== "all") {
    filtered = filtered.filter((item) => item.model == currentFilter);
  }

  renderHistory(filtered);
}

// ================= EVENTS =================

// Search
searchInput.addEventListener("input", applyFilters);

// Filter buttons
filterButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    filterButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");

    currentFilter = btn.dataset.filter;
    applyFilters();
  });
});

// ================= INIT =================
fetchHistory();

// ================= MENU =================
var menu = document.querySelector(".menu");
var menu_toggle = document.querySelector(".menu_toggle");

menu_toggle.onclick = function () {
  menu_toggle.classList.toggle("active");
  menu.classList.toggle("responsive");
};
