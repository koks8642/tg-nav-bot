/* RQM navigation Mini App — vanilla JS, no build step. */
"use strict";

const tg = window.Telegram ? window.Telegram.WebApp : null;
if (tg) { tg.ready(); tg.expand(); }

const initData = tg ? tg.initData : "";
const params = new URLSearchParams(location.search);
const ADMIN_REQUESTED = params.get("admin") === "1";

const view = document.getElementById("view");
const searchInput = document.getElementById("search");
const backBtn = document.getElementById("backBtn");
const adminBtn = document.getElementById("adminBtn");
const tabbar = document.getElementById("tabbar");

let history = [];          // route stack for the back button
let isOwner = false;

// ── api ──────────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  if (initData) headers["X-Telegram-Init-Data"] = initData;
  const res = await fetch(path, { headers, ...opts });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.status);
  return res.json();
}
const apiPost = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify({ ...body, initData }) });

function toast(msg) {
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 1800);
}
const esc = (s) => (s == null ? "" : String(s)).replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ── router ───────────────────────────────────────────────────────────────────
const routes = {};
function route(name, fn) { routes[name] = fn; }

async function go(name, arg, push = true) {
  if (push) history.push([name, arg]);
  backBtn.hidden = history.length <= 1;
  view.innerHTML = '<div class="loading">Загрузка…</div>';
  try { await routes[name](arg); }
  catch (e) { view.innerHTML = `<div class="empty">Ошибка: ${esc(e.message)}</div>`; }
  view.scrollTop = 0;
}
backBtn.onclick = () => {
  history.pop();
  const [name, arg] = history[history.length - 1] || ["home"];
  go(name, arg, false);
};
tabbar.querySelectorAll("button").forEach((b) => {
  b.onclick = () => {
    tabbar.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    history = [];
    go(b.dataset.route, null);
  };
});

// ── home ─────────────────────────────────────────────────────────────────────
route("home", async () => {
  const [{ projects }, { sections }] = await Promise.all([
    api("/api/projects"), api("/api/sections"),
  ]);
  let html = '<div class="section-title">Проекты</div>';
  html += projects.map((p) => `
    <a class="card row" data-go="project" data-arg="${p.id}">
      <span class="emoji">${esc(p.emoji)}</span>
      <span><div class="title">${esc(p.canonical_name)}</div>
      <div class="sub">${p.chapter_count} глав</div></span>
      <span class="badge">${p.chapter_count}</span>
    </a>`).join("") || '<div class="empty">Проектов пока нет</div>';

  html += '<div class="section-title">Разделы</div>';
  html += sections.map((s) => `
    <a class="card row" data-go="section" data-arg="${s.id}">
      <span class="emoji">${esc(s.emoji)}</span>
      <div class="title">${esc(s.name)}</div>
    </a>`).join("");
  view.innerHTML = html;
  bindGo();
});

// ── search (debounced) ───────────────────────────────────────────────────────
let searchTimer = null;
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchInput.value.trim();
  if (!q) { go("home", null); return; }
  searchTimer = setTimeout(() => runSearch(q), 180);
});

async function runSearch(q) {
  const { projects, chapters } = await api("/api/search?q=" + encodeURIComponent(q));
  let html = "";
  if (projects && projects.length) {
    html += '<div class="section-title">Проекты</div>';
    html += projects.map((p) => `
      <a class="card row" data-go="project" data-arg="${p.id}">
        <span class="emoji">${esc(p.emoji)}</span>
        <div class="title">${esc(p.canonical_name)}</div></a>`).join("");
  }
  html += '<div class="section-title">Главы</div>';
  html += (chapters || []).map(chapterRow).join("") ||
    '<div class="empty">Ничего не найдено</div>';
  view.innerHTML = html;
  bindGo();
}

function chapterRow(c) {
  const arc = c.arc ? `<div class="arc">${esc(c.project_name)} · ${esc(c.arc)}</div>` :
    `<div class="arc">${esc(c.project_name)}</div>`;
  const post = c.post_url ? `<a class="post" href="${esc(c.post_url)}" target="_blank">💬</a>` : "";
  return `<div class="chapter">
    <div class="num">${c.number}</div>
    <div class="meta"><div class="ttl">${esc(c.title || ("Глава " + c.number))}</div>${arc}</div>
    <a class="go" href="${esc(c.telegraph_url)}" target="_blank">Читать</a>${post}
  </div>`;
}

// ── project page ─────────────────────────────────────────────────────────────
route("project", async (id) => {
  const data = await api("/api/project/" + id);
  const p = data.project;
  const ext = data.external_links || [];
  const labels = { ranobelib: "📚 RanobeLib", mangalib: "🖼 MangaLib",
    senkuro: "🌸 Senkuro", boosty: "💎 Boosty" };
  const byPlatform = {};
  ext.forEach((e) => { if (!byPlatform[e.platform]) byPlatform[e.platform] = e.url; });

  let html = `<h2 class="proj"><span>${esc(p.emoji)}</span>${esc(p.canonical_name)}</h2>`;
  const extHtml = Object.keys(byPlatform).map((pl) =>
    `<a href="${esc(byPlatform[pl])}" target="_blank">${labels[pl] || pl}</a>`).join("");
  if (extHtml) html += `<div class="ext">${extHtml}</div>`;
  html += `<input class="filter" placeholder="Фильтр по этому проекту…" id="projFilter">`;
  html += `<div id="chList"></div>`;
  view.innerHTML = html;

  const chapters = data.chapters || [];
  const list = document.getElementById("chList");
  function renderList(filter) {
    const f = (filter || "").toLowerCase();
    const shown = f ? chapters.filter((c) =>
      String(c.number).includes(f) ||
      (c.arc || "").toLowerCase().includes(f) ||
      (c.title || "").toLowerCase().includes(f)) : chapters;
    list.innerHTML = renderGroupedByArc(shown);
  }
  renderList("");
  document.getElementById("projFilter").addEventListener("input", (e) =>
    renderList(e.target.value));

  if (isOwner) {
    const edit = document.createElement("button");
    edit.className = "btn secondary"; edit.textContent = "🛠 Редактировать проект";
    edit.onclick = () => go("admin_project", p);
    view.appendChild(edit);
  }
});

function renderGroupedByArc(chapters) {
  const groups = [];
  const idx = {};
  chapters.forEach((c) => {
    const arc = c.arc || "Без арки";
    if (idx[arc] === undefined) { idx[arc] = groups.length; groups.push([arc, []]); }
    groups[idx[arc]][1].push(c);
  });
  return groups.map(([arc, cs]) =>
    `<div class="arc-head">📂 ${esc(arc)}</div>` + cs.map(chapterRow).join("")
  ).join("") || '<div class="empty">Глав нет</div>';
}

// ── section page ─────────────────────────────────────────────────────────────
route("section", async (id) => {
  const { section, items } = await api("/api/section/" + id);
  let html = `<h2 class="proj"><span>${esc(section.emoji)}</span>${esc(section.name)}</h2>`;
  if (!items.length) html += '<div class="empty">— пока пусто —</div>';
  else html += items.map((it) => `
    <a class="card" href="${esc(it.url)}" target="_blank">
      <div class="title">${esc(it.title || "Без названия")}</div>
      ${it.date ? `<div class="sub">${esc(it.date.slice(0, 10))}</div>` : ""}
    </a>`).join("");
  view.innerHTML = html;
});

// ── activity feed ────────────────────────────────────────────────────────────
route("activity", async () => {
  const { activity } = await api("/api/activity");
  view.innerHTML = '<div class="section-title">Последние публикации</div>' +
    (activity.map((a) => `
      <a class="card row" href="${esc(a.telegraph_url)}" target="_blank">
        <span class="emoji">${esc(a.project_emoji)}</span>
        <span><div class="title">${esc(a.project_name)} — Глава ${a.number}</div>
        <div class="sub">${esc(a.arc || "")}</div></span>
      </a>`).join("") || '<div class="empty">Пока пусто</div>');
});

// ── admin ────────────────────────────────────────────────────────────────────
route("admin", async () => {
  const data = await api("/api/admin/lists");
  let html = `<div class="pill-tabs">
    <button class="active" data-atab="projects">Проекты</button>
    <button data-atab="hashtags">Хэштеги</button>
    <button data-atab="conflicts">Конфликты</button>
    <button data-atab="ops">Действия</button></div><div id="adminBody"></div>`;
  view.innerHTML = html;
  const body = document.getElementById("adminBody");
  const tabs = view.querySelectorAll(".pill-tabs button");
  tabs.forEach((t) => t.onclick = () => {
    tabs.forEach((x) => x.classList.remove("active")); t.classList.add("active");
    renderAdminTab(t.dataset.atab, body, data);
  });
  renderAdminTab("projects", body, data);
});

function renderAdminTab(tab, body, data) {
  if (tab === "projects") {
    body.innerHTML = data.projects.map((p) => `
      <div class="card row" data-go="admin_project" data-arg='${p.id}'>
        <span class="emoji">${esc(p.emoji)}</span>
        <span><div class="title">${esc(p.canonical_name)}</div>
        <div class="sub">${p.chapter_count} глав${p.hidden ? " · скрыт" : ""}</div></span>
      </div>`).join("");
    body.querySelectorAll("[data-go]").forEach((el) => {
      el.onclick = () => {
        const p = data.projects.find((x) => x.id == el.dataset.arg);
        go("admin_project", p);
      };
    });
  } else if (tab === "hashtags") {
    body.innerHTML = data.hashtags.map((h) =>
      `<div class="card"><div class="title">#${esc(h.hashtag)}</div>
       <div class="sub">${esc(h.kind)} → id ${h.target_id}</div></div>`).join("") +
      hashtagForm(data);
    bindHashtagForm(data);
  } else if (tab === "conflicts") {
    api("/api/admin/conflicts").then(({ conflicts }) => {
      body.innerHTML = conflicts.length ? conflicts.map((c) => `
        <div class="card"><div class="title">[${esc(c.type)}] ${esc(c.ref || "")}</div>
        <div class="sub">${esc(c.detail)}</div>
        <button class="btn secondary" data-resolve="${c.id}">✓ Решить</button></div>`).join("")
        : '<div class="empty">Конфликтов нет</div>';
      body.querySelectorAll("[data-resolve]").forEach((b) => b.onclick = async () => {
        await apiPost("/api/admin/conflicts", { id: +b.dataset.resolve, status: "resolved" });
        toast("Решено"); renderAdminTab("conflicts", body, data);
      });
    });
  } else if (tab === "ops") {
    body.innerHTML = `
      <button class="btn" id="opRebuild">♻️ Пересобрать все страницы</button>
      <button class="btn secondary" id="opBackfill">📥 Перезапустить бэкафилл</button>`;
    document.getElementById("opRebuild").onclick = async () => {
      await apiPost("/api/admin/rebuild", {}); toast("Пересборка в очереди");
    };
    document.getElementById("opBackfill").onclick = async () => {
      toast("Бэкафилл запущен…");
      const r = await apiPost("/api/admin/backfill", {}); toast("Готово");
    };
  }
}

function hashtagForm(data) {
  return `<div class="admin-form card">
    <div class="title">Привязать хэштег</div>
    <label>Хэштег (без #)</label><input id="htTag">
    <label>Тип</label><select id="htKind"><option value="project">project</option>
      <option value="category">category</option></select>
    <label>Цель</label><select id="htTarget"></select>
    <button class="btn" id="htSave">Сохранить</button></div>`;
}
function bindHashtagForm(data) {
  const kind = document.getElementById("htKind");
  const target = document.getElementById("htTarget");
  function fill() {
    const list = kind.value === "project" ? data.projects : data.sections;
    target.innerHTML = list.map((x) =>
      `<option value="${x.id}">${esc(x.canonical_name || x.name)}</option>`).join("");
  }
  kind.onchange = fill; fill();
  document.getElementById("htSave").onclick = async () => {
    const tag = document.getElementById("htTag").value.trim();
    if (!tag) return;
    await apiPost("/api/admin/hashtag", { hashtag: tag, kind: kind.value,
      target_id: +target.value });
    toast("Сохранено");
  };
}

route("admin_project", async (p) => {
  view.innerHTML = `<div class="admin-form">
    <h2 class="proj">${esc(p.emoji)} ${esc(p.canonical_name)}</h2>
    <label>Название</label><input id="f_name" value="${esc(p.canonical_name)}">
    <label>Эмодзи</label><input id="f_emoji" value="${esc(p.emoji)}">
    <label>RanobeLib URL</label><input id="f_rl" value="${esc(p.ranobelib_url)}">
    <label>MangaLib URL</label><input id="f_ml" value="${esc(p.mangalib_url)}">
    <label>Senkuro URL</label><input id="f_sk" value="${esc(p.senkuro_url)}">
    <label>Boosty URL</label><input id="f_bo" value="${esc(p.boosty_url)}">
    <label>Порядок</label><input id="f_sort" type="number" value="${p.sort_order}">
    <label><input type="checkbox" id="f_hidden" ${p.hidden ? "checked" : ""}> Скрыть проект</label>
    <button class="btn" id="saveP">💾 Сохранить</button>
  </div>`;
  document.getElementById("saveP").onclick = async () => {
    await apiPost("/api/admin/project", {
      id: p.id,
      canonical_name: document.getElementById("f_name").value,
      emoji: document.getElementById("f_emoji").value,
      ranobelib_url: document.getElementById("f_rl").value,
      mangalib_url: document.getElementById("f_ml").value,
      senkuro_url: document.getElementById("f_sk").value,
      boosty_url: document.getElementById("f_bo").value,
      sort_order: +document.getElementById("f_sort").value,
      hidden: document.getElementById("f_hidden").checked ? 1 : 0,
    });
    toast("Сохранено и поставлено в пересборку");
  };
});

// ── delegated navigation for data-go cards ───────────────────────────────────
function bindGo() {
  view.querySelectorAll("[data-go]").forEach((el) => {
    el.onclick = (e) => {
      if (e.target.tagName === "A" && e.target.classList.contains("go")) return;
      go(el.dataset.go, el.dataset.arg);
    };
  });
}

// ── boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  if (initData) {
    try {
      const who = await api("/api/admin/whoami");
      isOwner = !!who.is_owner;
    } catch { isOwner = false; }
  }
  if (isOwner) {
    adminBtn.hidden = false;
    adminBtn.onclick = () => { history = []; go("admin", null); };
    if (ADMIN_REQUESTED) { go("admin", null); return; }
  }
  go("home", null);
}
boot();
