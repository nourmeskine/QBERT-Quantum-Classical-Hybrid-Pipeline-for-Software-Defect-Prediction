// menu responsive code
var menu = document.querySelector(".menu");
var menu_toggle = document.querySelector(".menu_toggle");

menu_toggle.onclick = function () {
  menu_toggle.classList.toggle("active");
  menu.classList.toggle("responsive");
};

//site animation

const header = document.querySelector("header");
const title_span = document.querySelectorAll(".left h1 span");
const p = document.querySelector(".left p");
const a = document.querySelector(".left a");
const img = document.querySelector(".image");

window.addEventListener("load", () => {
  const TL = gsap.timeline({ paused: true });
  TL.staggerFrom(header, 2, { y: -100, opacity: 0, ease: "power2.out" }, 0.1)
    .staggerFrom(img, 1, { x: 1000, opacity: 0, ease: "power2.out" }, 0.3)
    .staggerFrom(title_span, 1, { opacity: 0, ease: "power2.out" }, 0.1)
    .staggerFrom(p, 1, { opacity: 0, ease: "power2.out" }, 0.2)
    .staggerFrom(a, 1, { opacity: 0, ease: "power2.out" }, 0.3);

  TL.play();
});
const aboutLink = document.querySelector(".about_link");

aboutLink.addEventListener("click", (e) => {
  e.preventDefault();
  document.getElementById("about").scrollIntoView({
    behavior: "smooth",
  });
});
// Scroll fluide vers Home
const homeLink = document.querySelector("header .menu li a[href='#']");

homeLink.addEventListener("click", (e) => {
  e.preventDefault(); // empêche le "tac tac" par défaut
  window.scrollTo({
    top: 0, // remonte au début de la page
    behavior: "smooth", // scroll fluide
  });
});
// Animation des modèles About au scroll
const models = document.querySelectorAll(".model");

function revealModels() {
  const triggerBottom = window.innerHeight * 0.9; // déclenchement à 90% de la hauteur

  models.forEach((model) => {
    const modelTop = model.getBoundingClientRect().top;

    if (modelTop < triggerBottom) {
      model.style.opacity = "1";
      model.style.transform = "translateY(0)";
    }
  });
}

// appeler au scroll et au chargement
window.addEventListener("scroll", revealModels);
window.addEventListener("load", revealModels);
// Modal Login pour tous les éléments qui contiennent "Login" dans le texte
const allLinks = document.querySelectorAll("a, button"); // tous les liens et boutons
const loginModal = document.getElementById("loginModal");
const closeModal = document.querySelector(".close_modal");

allLinks.forEach((el) => {
  if (el.textContent.trim().toLowerCase() === "login") {
    // texte exactement "login"
    el.addEventListener("click", (e) => {
      e.preventDefault(); // empêcher le lien par défaut
      loginModal.style.display = "flex";
    });
  }
});

// fermer le modal avec X
closeModal.addEventListener("click", () => {
  loginModal.style.display = "none";
});

// fermer modal si clic en dehors du rectangle
window.addEventListener("click", (e) => {
  if (e.target === loginModal) {
    loginModal.style.display = "none";
  }
});

// Form Login ↔ Sign Up
const formLogin = document.querySelector(".auth_login_form");
const formSignup = document.querySelector(".auth_signup_form");

// lien Sign Up dans le texte
const showSignup = document.querySelector(".show_signup");
// lien Login dans Sign Up
const showLogin = document.querySelector(".show_login");

if (showSignup) {
  showSignup.addEventListener("click", (e) => {
    e.preventDefault();
    formLogin.style.display = "none";
    formSignup.style.display = "block";
  });
}

if (showLogin) {
  showLogin.addEventListener("click", (e) => {
    e.preventDefault();
    formSignup.style.display = "none";
    formLogin.style.display = "block";
  });
}
const aboutSection = document.querySelector(".about_models");

function revealAbout() {
  const top = aboutSection.getBoundingClientRect().top;
  const trigger = window.innerHeight * 0.8;

  if (top < trigger) {
    aboutSection.classList.add("show");
  }
}
//passage de login vers debug
window.addEventListener("scroll", revealAbout);
document
  .querySelector(".auth_login_form form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const email = this.querySelector("input[type='email']").value;
    const password = this.querySelector("input[type='password']").value;

    const res = await fetch("http://127.0.0.1:8001/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();

    if (data.success) {
      window.location.href = "index.html"; //  go to debug
    } else {
      alert("Invalid login");
    }
  });

// passage de signup vers debug
document
  .querySelector(".auth_signup_form form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    const inputs = this.querySelectorAll("input");

    const name = inputs[0].value;
    const email = inputs[1].value;
    const password = inputs[2].value;
    const confirm = inputs[3].value;

    if (password !== confirm) {
      alert("Passwords do not match");
      return;
    }

    const res = await fetch("http://127.0.0.1:8001/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });

    const data = await res.json();

    if (data.success) {
      window.location.href = "index.html"; //  go to debug
    } else {
      alert("Signup failed");
    }
  });

function goToDebug() {
  // Optionnel : tu peux ajouter une petite alerte ou un log pour confirmer
  console.log("Direction : Page de Debugging...");

  // La commande magique pour changer de page
  window.location.href = "debug.html";
}
