/*
 * Grounded embeddable chat widget.
 *
 * Usage on any third-party site:
 *   <script src="https://app.example.com/widget.js"
 *           data-project-key="pk_xxx"
 *           data-api-base="https://api.example.com"></script>
 *
 * Renders a floating chat bubble in a Shadow DOM (so host-page CSS can't leak
 * in) and talks to the public, key-scoped chat API. No user auth required.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  var projectKey = script && script.getAttribute("data-project-key");
  var apiBase =
    (script && script.getAttribute("data-api-base")) || window.location.origin;
  var title = (script && script.getAttribute("data-title")) || "Ask a question";

  if (!projectKey) {
    console.error("[grounded] missing data-project-key on <script> tag");
    return;
  }

  var sessionId =
    "w_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

  var host = document.createElement("div");
  host.style.position = "fixed";
  host.style.bottom = "20px";
  host.style.right = "20px";
  host.style.zIndex = "2147483647";
  document.body.appendChild(host);
  var root = host.attachShadow({ mode: "open" });

  root.innerHTML =
    '<style>' +
    ":host{all:initial}" +
    "*{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}" +
    ".bubble{width:56px;height:56px;border-radius:50%;background:#4f46e5;color:#fff;" +
    "border:none;cursor:pointer;font-size:24px;box-shadow:0 4px 14px rgba(0,0,0,.25)}" +
    ".panel{display:none;flex-direction:column;width:360px;max-width:90vw;height:520px;" +
    "max-height:75vh;background:#fff;border-radius:12px;overflow:hidden;" +
    "box-shadow:0 12px 40px rgba(0,0,0,.25)}" +
    ".panel.open{display:flex}" +
    ".hdr{background:#4f46e5;color:#fff;padding:12px 16px;font-weight:600}" +
    ".log{flex:1;overflow-y:auto;padding:12px;background:#f8fafc}" +
    ".msg{margin:8px 0;padding:8px 12px;border-radius:10px;max-width:85%;white-space:pre-wrap;line-height:1.4}" +
    ".user{background:#4f46e5;color:#fff;margin-left:auto}" +
    ".bot{background:#fff;border:1px solid #e2e8f0;color:#0f172a}" +
    ".src{font-size:11px;color:#64748b;margin-top:4px}" +
    ".row{display:flex;border-top:1px solid #e2e8f0}" +
    ".row input{flex:1;border:none;padding:12px;font-size:14px;outline:none}" +
    ".row button{border:none;background:#4f46e5;color:#fff;padding:0 16px;cursor:pointer}" +
    "</style>" +
    '<button class="bubble" aria-label="Open chat">\u{1F4AC}</button>' +
    '<div class="panel" role="dialog">' +
    '  <div class="hdr"></div>' +
    '  <div class="log"></div>' +
    '  <div class="row"><input type="text" placeholder="Type your question..."/>' +
    '  <button class="send">Send</button></div>' +
    "</div>";

  var bubble = root.querySelector(".bubble");
  var panel = root.querySelector(".panel");
  var log = root.querySelector(".log");
  var input = root.querySelector("input");
  var send = root.querySelector(".send");
  root.querySelector(".hdr").textContent = title;

  bubble.addEventListener("click", function () {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) input.focus();
  });

  function addMessage(text, who, sources) {
    var el = document.createElement("div");
    el.className = "msg " + who;
    el.textContent = text;
    if (sources && sources.length) {
      var s = document.createElement("div");
      s.className = "src";
      s.textContent = "Sources: " + sources.length + " passage(s)";
      el.appendChild(s);
    }
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  function ask(question) {
    addMessage(question, "user");
    input.value = "";
    var thinking = document.createElement("div");
    thinking.className = "msg bot";
    thinking.textContent = "…";
    log.appendChild(thinking);

    fetch(apiBase + "/api/v1/public/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        public_key: projectKey,
        question: question,
        session_id: sessionId,
      }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        thinking.remove();
        addMessage(data.answer, "bot", data.sources);
      })
      .catch(function () {
        thinking.remove();
        addMessage("Sorry, something went wrong. Please try again.", "bot");
      });
  }

  function submit() {
    var q = input.value.trim();
    if (q) ask(q);
  }
  send.addEventListener("click", submit);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") submit();
  });
})();
