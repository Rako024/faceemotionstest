/* Bütün pəncərələr üçün ortaq: Python körpüsü, ikonlar, köməkçilər. */
"use strict";

const BOOT = window.__BOOT__ || {};

if (BOOT.material && BOOT.material !== "none") document.documentElement.classList.add("material");

/* Python API (pywebview). Brauzerdə önizləmə üçün mock.js MockApi təqdim edir. */
const Api = (() => {
  const ready = new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve(window.pywebview.api);
    window.addEventListener("pywebviewready", () => resolve(window.pywebview.api), { once: true });
    setTimeout(() => { if (!window.pywebview && window.MockApi) resolve(window.MockApi); }, 250);
  });
  return {
    ready,
    async call(name, ...args) {
      const api = await ready;
      try {
        return await api[name](...args);
      } catch (err) {
        console.error(name, err);
        return null;
      }
    },
  };
})();

/* SF Symbols üslubunda xətti ikonlar (24×24) */
const ICONS = {
  face: '<circle cx="12" cy="12" r="9"/><path d="M9 10h.01M15 10h.01"/><path d="M9.5 15a3.5 3.5 0 0 0 5 0"/>',
  "face-off": '<circle cx="12" cy="12" r="9"/><path d="M9 10h.01M15 10h.01M9.5 15.5h5"/><path d="M5 5l14 14"/>',
  scan: '<path d="M4 8V6a2 2 0 0 1 2-2h2M4 16v2a2 2 0 0 0 2 2h2M16 4h2a2 2 0 0 1 2 2v2M16 20h2a2 2 0 0 0 2-2v-2"/><path d="M9 10h.01M15 10h.01"/><path d="M9.5 15a3.5 3.5 0 0 0 5 0"/>',
  database: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6"/><path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/>',
  "wifi-off": '<path d="M12 18h.01"/><path d="M9.17 15.17a4 4 0 0 1 5.66 0"/><path d="M6.34 12.34a8 8 0 0 1 3.4-2.03M14.3 10.3a8 8 0 0 1 3.36 2.04"/><path d="M3.5 9.5a12 12 0 0 1 3.1-2.2M11 6.05A12 12 0 0 1 20.5 9.5"/><path d="M3 3l18 18"/>',
  hand: '<path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M11 5.5v-1a1.5 1.5 0 0 1 3 0V12"/><path d="M14 5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M17 7.5a1.5 1.5 0 0 1 3 0V16a6 6 0 0 1-6 6h-1.8a6 6 0 0 1-5-2.7L7 19c-.3-.5-1.4-2.4-3.3-5.7a1.5 1.5 0 0 1 .5-2 1.9 1.9 0 0 1 2.3.3L8 13"/>',
  bubble: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.3A8 8 0 1 1 21 12z"/>',
  video: '<path d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14"/><rect x="3" y="6" width="12" height="12" rx="2.5"/>',
  "video-off": '<path d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14"/><path d="M10 6h3a2 2 0 0 1 2 2v3m0 4v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2"/><path d="M3 3l18 18"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="3"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  x: '<path d="M17 7L7 17M7 7l10 10"/>',
  minus: '<path d="M6 12h12"/>',
  expand: '<path d="M8 4H4v4M16 20h4v-4M4 4l6 6M20 20l-6-6"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  cpu: '<rect x="5" y="5" width="14" height="14" rx="3"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M3 10h2M3 14h2M10 3v2M14 3v2M21 10h-2M21 14h-2M14 21v-2M10 21v-2"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  bolt: '<path d="M13 3L4 14h7l-1 7 9-11h-7l1-7z"/>',
  chevron: '<path d="M6 9l6 6 6-6"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  refresh: '<path d="M20 11A8.1 8.1 0 0 0 4.5 9M4 5v4h4"/><path d="M4 13a8.1 8.1 0 0 0 15.5 2m.5 4v-4h-4"/>',
  shield: '<path d="M12 3a12 12 0 0 0 8.5 3A12 12 0 0 1 12 21 12 12 0 0 1 3.5 6 12 12 0 0 0 12 3"/><path d="M9 12l2 2 4-4"/>',
  file: '<path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z"/>',
  play: '<path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/>',
  star: '<path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z"/>',
};

function icon(name, cls = "") {
  return `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;
}

/* data-icon="ad" olan elementlərə ikon yerləşdirir */
function mountIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon, el.dataset.iconClass || ""));
    el.removeAttribute("data-icon");
  });
}

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function setText(el, value) {
  if (el && el.textContent !== String(value)) el.textContent = value;
}

/* 12345 → "12 345" (Azərbaycan dilində minliklər boşluqla ayrılır) */
function fmtInt(n) {
  return String(Math.round(n || 0)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

function pct(p, digits = 0) {
  return `${(p * 100).toFixed(digits)}%`;
}

function mmss(seconds) {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/* Rəqəmi yumşaq şəkildə dəyişdirir (iOS-dakı kimi) */
function tweenNumber(el, to, format, duration = 450) {
  const from = parseFloat(el.dataset.value || "0");
  el.dataset.value = String(to);
  if (Math.abs(from - to) < 1e-4 || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.textContent = format(to);
    return;
  }
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / duration);
    const e = 1 - Math.pow(1 - k, 3);
    el.textContent = format(from + (to - from) * e);
    if (k < 1 && el.dataset.value === String(to)) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* Emosiyaya görə ambient fon rəngləri */
const AMBIENT = {
  Anger: ["--red", "--orange"],
  Contempt: ["--purple", "--indigo"],
  Disgust: ["--green", "--teal"],
  Fear: ["--indigo", "--purple"],
  Happiness: ["--orange", "--yellow"],
  Neutral: ["--teal", "--blue"],
  Sadness: ["--blue", "--indigo"],
  Surprise: ["--yellow", "--orange"],
};

function setAmbient(emotion) {
  const root = document.documentElement.style;
  const pair = AMBIENT[emotion];
  if (!pair) {
    ["--ambient-a", "--ambient-b", "--ambient-c"].forEach((v) => root.removeProperty(v));
    return;
  }
  root.setProperty("--ambient-a", `color-mix(in srgb, var(${pair[0]}) 55%, transparent)`);
  root.setProperty("--ambient-b", `color-mix(in srgb, var(${pair[1]}) 45%, transparent)`);
}

document.addEventListener("contextmenu", (e) => e.preventDefault());
