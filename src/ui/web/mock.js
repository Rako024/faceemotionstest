/* YALNIZ brauzerdə dizayn önizləməsi üçün saxta API. Python bu faylı səhifəyə daxil etmir.
   URL parametrləri: ?phase=running|consent|stopped|finished|error  &blind=1  &mtl=1  &noface=1  &uncertain=1
                     &attire=formal|not|unknown|error|off */
"use strict";

(() => {
  if (window.pywebview) return;
  const q = new URLSearchParams(location.search);
  const CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"];
  const LABELS = { Anger: "Qəzəb", Contempt: "Həqarət", Disgust: "İkrah", Fear: "Qorxu", Happiness: "Xoşbəxtlik", Neutral: "Neytral", Sadness: "Kədər", Surprise: "Təəccüb" };
  const mtl = q.get("mtl") === "1";

  window.__BOOT__ = Object.assign(
    {
      operator_id: "OP-7F3A",
      compare_label: "Avto — 4 modelə qədər, gücə görə",
      model: mtl ? "enet_b0_8_va_mtl" : "enet_b0_8_best_afew",
      model_short: mtl ? "enet_b0 · VA" : "enet_b0 · afew",
      engine_label: "ONNX · CPU",
      analysis_fps: 1.5,
      window_seconds: 10,
      min_frames: 3,
      source: "camera",
      prompts_enabled: true,
      prompt_interval_min: 10,
      prompt_timeout_s: 30,
      show_prediction: q.get("blind") !== "1",
      preview_fps: 6,
      is_mtl: mtl,
      material: "none",
      classes: CLASSES.map((k) => ({ key: k, label: LABELS[k] })),
      categories: [
        { code: "neutral", label: "Neytral" }, { code: "happy", label: "Yaxşı / Xoşbəxt" },
        { code: "tired", label: "Yorğun" }, { code: "angry", label: "Əsəbi" },
        { code: "sad", label: "Kədərli" }, { code: "stressed", label: "Stresli" },
      ],
      decline_label: "Cavab vermək istəmirəm",
      initial_phase: q.get("phase") || "consent",
      attire_enabled: q.get("attire") !== "off",
      attire_frames: 5,
      uncertain_threshold: 0.3,
      research_enabled: q.get("research") !== "0",
      test_expressions: 7,
      test_rounds: 2,
      test_minutes: 3,
    },
    window.__BOOT__ || {}
  );
  Object.assign(BOOT, window.__BOOT__);

  let phase = window.__BOOT__.initial_phase;
  let frames = phase === "running" ? 14 : 0;
  let preview = false;
  let promptId = q.get("prompt") === "1" ? 1 : 0;
  let promptShown = performance.now();
  const t0 = performance.now();

  function probs(shift = 0, fer = false) {
    const t = (performance.now() - t0) / 1000 + shift * 7;
    const raw = CLASSES.map((c, i) => Math.exp(Math.sin(t / 3 + i + shift) * 0.6 + (c === "Neutral" ? 2.2 - shift * 0.4 : c === "Happiness" ? 1.2 + Math.sin(t / 5) + shift * 0.3 : 0)));
    if (q.get("uncertain") === "1") raw.forEach((_, i) => (raw[i] = 1 + i * 0.05));
    if (fer) raw[1] = 0;
    const s = raw.reduce((a, b) => a + b, 0);
    return raw.map((v) => v / s);
  }

  const MODELS = [
    { key: "enet_b0_8_best_afew", label: "enet_b0 · afew", dataset: "AffectNet", family: "emotiefflib", ms: 12, shift: 0 },
    { key: "enet_b0_8_va_mtl", label: "enet_b0 · VA", dataset: "AffectNet", family: "emotiefflib", ms: 12, shift: 0.3, va: true },
    { key: "enet_b2_8", label: "enet_b2", dataset: "AffectNet", family: "emotiefflib", ms: 22, shift: 0.6 },
    { key: "mbf_va_mtl", label: "mbf · VA", dataset: "AffectNet", family: "emotiefflib", ms: 7, shift: 0.9, va: true },
    { key: "deepface_emotion", label: "DeepFace", dataset: "FER2013", family: "fer", ms: 0.6, shift: 1.3 },
    { key: "vit_fer2013", label: "ViT (HF)", dataset: "FER2013", family: "fer", ms: 144, shift: 2.1 },
    { key: "kaggle_fer_cnn", label: "Kaggle CNN", dataset: "FER2013", family: "fer", ms: 3, shift: 1.7, missing: true },
  ];
  let primaryKey = "enet_b0_8_best_afew";
  let compareMode = q.get("compare") || "auto";
  const enabled = { enet_b0_8_best_afew: true, deepface_emotion: true, vit_fer2013: true };
  const history = {};

  function modelResult(m, noface) {
    const p = probs(m.shift, m.family === "fer");
    const top = p.indexOf(Math.max(...p));
    const status = noface ? "no_face" : frames < 3 ? "warming_up" : p[top] < BOOT.uncertain_threshold ? "uncertain" : "ok";
    const result = { status, status_label: status, n_frames: Math.min(frames, 15), window_s: 10 };
    if (BOOT.show_prediction && (status === "ok" || status === "uncertain")) {
      Object.assign(result, {
        top: CLASSES[top], top_label: LABELS[CLASSES[top]], top_prob: p[top],
        probs: CLASSES.map((k, i) => ({ key: k, label: LABELS[k], p: m.family === "fer" && k === "Contempt" ? null : p[i] })),
        valence: m.va || (mtl && m.key === primaryKey) ? 0.12 + Math.sin(frames / 6) * 0.3 : null,
        arousal: m.va || (mtl && m.key === primaryKey) ? -0.08 + Math.cos(frames / 7) * 0.25 : null,
      });
    }
    return result;
  }

  // Nəzarətli test simulyasiyası (?test=run | ?test=done)
  const TEST_EXPR = ["Neutral", "Happiness", "Sadness", "Surprise", "Anger", "Fear", "Disgust"];
  const TEST_STEPS = [...TEST_EXPR, "Neutral", ...TEST_EXPR.slice(1).reverse()];
  const TD = { intro: 4, prepare: 3, hold: 8, rest: 2 };
  let testStart = q.get("test") === "run" ? performance.now() : q.get("test") === "done" ? -1e9 : null;
  let testCancelled = false;
  let testTester = "T-01";
  let testSave = q.get("save") === "1";
  const CAMS = [{ index: 0, name: "HP HD Camera" }, { index: 1, name: "Logi C270 HD WebCam" }];
  let camIndex = Number(q.get("cam") || 0);
  const HINTS = { Neutral: "Sakit, ifadəsiz baxın — üz əzələləri rahat", Happiness: "Gülümsəyin — yanaqlar qalxsın, dişlər görünə bilər",
    Sadness: "Kədərli görünün — dodaq kənarları aşağı, qaşların içi yuxarı", Surprise: "Təəccüblənin — qaşlar yuxarı, gözlər geniş, ağız açıq",
    Anger: "Qəzəbli görünün — qaşlar çatılıb aşağı, dodaqlar sıxılıb", Fear: "Qorxmuş görünün — gözlər geniş açıq, qaşlar yuxarı",
    Disgust: "İkrah — burnu qırışdırın, üst dodaq yuxarı" };

  function testState() {
    if (testStart === null) return null;
    let t = (performance.now() - testStart) / 1000;
    const stepLen = TD.prepare + TD.hold + TD.rest;
    const total = TD.intro + stepLen * TEST_STEPS.length;
    const saved = testSave ? Math.min(160, Math.round(t * 1.2)) : null;
    const base = { total: TEST_STEPS.length, rounds: 2, steps: TEST_STEPS, cancelled: testCancelled, tester_id: testTester,
      saved_frames: saved, dataset_session: testSave ? `20261022_101500_${testTester}` : null };
    if (testCancelled || t >= total) {
      const accs = { enet_b0_8_best_afew: [0.62, 0.95, 0.48, 0.71, 0.66, 0.22, 0.41], enet_b0_8_va_mtl: [0.58, 0.93, 0.52, 0.66, 0.70, 0.30, 0.55],
        deepface_emotion: [0.31, 0.98, 0.20, 0.55, 0.28, 0.12, 0.05], vit_fer2013: [0.71, 0.90, 0.35, 0.75, 0.40, 0.18, 0.10] };
      const summary = Object.entries(accs).map(([key, a]) => {
        const m = MODELS.find((x) => x.key === key);
        const bal = a.reduce((s, v) => s + v, 0) / a.length;
        return { key, label: m.label, dataset: m.dataset, frames: 168, face: 160, accuracy: bal + 0.04, balanced: bal, uncertain: 0.3 + Math.random() * 0.2,
          per_expression: Object.fromEntries(TEST_EXPR.map((e, i) => [e, a[i]])) };
      }).sort((x, y) => y.balanced - x.balanced);
      return { ...base, active: false, done: true, phase: "done", summary, expressions: TEST_EXPR.map((k) => ({ key: k, label: LABELS[k] })) };
    }
    if (t < TD.intro) return { ...base, active: true, done: false, phase: "intro", seconds_left: TD.intro - t, phase_seconds: TD.intro, step: 0, round: 1, expected: "Neutral", label: LABELS.Neutral, hint: HINTS.Neutral };
    t -= TD.intro;
    const i = Math.floor(t / stepLen);
    let w = t - i * stepLen;
    const e = TEST_STEPS[i];
    const common = { ...base, active: true, done: false, step: i, round: i < 7 ? 1 : 2, expected: e, label: LABELS[e], hint: HINTS[e] };
    if (w < TD.prepare) return { ...common, phase: "prepare", seconds_left: TD.prepare - w, phase_seconds: TD.prepare };
    w -= TD.prepare;
    if (w < TD.hold) return { ...common, phase: "hold", seconds_left: TD.hold - w, phase_seconds: TD.hold };
    return { ...common, phase: "rest", seconds_left: TD.rest - (w - TD.hold), phase_seconds: TD.rest };
  }

  // Geyim: yükləmə (yalnız ilk dəfə) → 5 kadr → nəticə
  const ATTIRE_KIND = q.get("attire") || "formal";
  const G = (shirt, polo, tshirt, knit, other) => [
    { key: "shirt", label: "düyməli köynək", p: shirt, formal: true }, { key: "polo", label: "polo", p: polo, formal: false },
    { key: "tshirt", label: "futbolka", p: tshirt, formal: false }, { key: "knitwear", label: "sviter / hudi", p: knit, formal: false },
    { key: "other", label: "digər geyim", p: other, formal: false },
  ].sort((a, b) => b.p - a.p);
  const ATTIRE_RESULTS = {
    formal: { result: "formal", label: "Rəsmi", reason: "", basis: "düyməli köynək + pencək", hint: null, garment: "düyməli köynək", color: "boz",
      garments: G(0.82, 0.07, 0.05, 0.04, 0.02), colors: [{ label: "boz", p: 0.48 }, { label: "naxışlı", p: 0.21 }, { label: "ağ", p: 0.12 }],
      tie: false, tie_p: 0.18, jacket: true, jacket_p: 0.81 },
    not: { result: "not_formal", label: "Qeyri-rəsmi", reason: "düyməli köynək və ya pencək görünmür — sviter / hudi", basis: "", hint: null,
      garment: "sviter / hudi", color: "boz", garments: G(0.06, 0.04, 0.12, 0.71, 0.07),
      colors: [{ label: "boz", p: 0.55 }, { label: "açıq mavi", p: 0.2 }, { label: "tünd", p: 0.1 }], tie: false, tie_p: 0.09, jacket: false, jacket_p: 0.14 },
    unknown: { result: "unknown", label: "Qiymətləndirilə bilmir", reason: "gövdə kadra kifayət qədər düşmür", basis: "",
      hint: "Kameranı bir az aşağı əyin və ya bir az geri oturun — yaxa görünməlidir", garment: null, color: null, garments: [], colors: [], tie: false, jacket: false },
  };
  let attireStart = window.__BOOT__.initial_phase === "running" ? performance.now() : null;
  let attireLoaded = false;
  let attireLast = null;
  function attireState() {
    if (!window.__BOOT__.attire_enabled || phase !== "running") return null;
    const base = { status: "idle", progress: 0, frames: 5, loaded: attireLoaded, error: null, result: attireLast };
    if (attireStart == null) return base;
    const t = (performance.now() - attireStart) / 1000;
    const loadS = attireLoaded ? 0 : 3;
    if (t < loadS) return { ...base, status: "loading" };
    const k = Math.floor((t - loadS) / 0.7) + 1;
    if (k <= 5) return { ...base, status: "checking", progress: k };
    attireLoaded = true;
    if (ATTIRE_KIND === "error") return { ...base, loaded: false, status: "error", error: "SigLIP 2 modeli tapılmadı — scripts/download_models.py --attire" };
    const r = ATTIRE_RESULTS[ATTIRE_KIND] || ATTIRE_RESULTS.formal;
    attireLast = { ...r, votes: r.result === "unknown" ? 0 : 4, age_s: Math.round(t - loadS - 3.5), formal_p: 0.9, white_p: 0.7 };
    return { ...base, status: "done", loaded: true, result: attireLast };
  }

  window.MockApi = {
    check_attire() {
      attireStart = performance.now();
      return Promise.resolve({ ok: true, error: null });
    },
    start_test(tester, save) {
      if (!/^[A-Za-z0-9_-]{2,32}$/.test(tester || "") || !/\d/.test(tester)) {
        return Promise.resolve({ ok: false, error: "Tester ID anonim olmalıdır: 2–32 simvol, ən azı bir rəqəm (məs. T-01)" });
      }
      testStart = performance.now(); testCancelled = false; preview = true; testTester = tester; testSave = !!save && BOOT.research_enabled;
      return Promise.resolve({ ok: true, error: null });
    },
    cancel_test() { testCancelled = true; return Promise.resolve(true); },
    dismiss_test() { testStart = null; return Promise.resolve(true); },
    state() {
      if (phase === "running") frames += 1;
      const noface = q.get("noface") === "1";
      const models = MODELS.map((m) => {
        const on = !m.missing && (m.key === primaryKey || (compareMode !== "off" && !!enabled[m.key]));
        const r = phase === "running" && on ? modelResult(m, noface) : null;
        let instant = null;
        if (r && r.status === "no_face") instant = { status: "no_face" };
        else if (r && r.top) {
          const top = Math.random() < 0.75 ? r.top : CLASSES[Math.floor(Math.random() * CLASSES.length)];
          const p = 0.2 + Math.random() * 0.75;
          instant = { status: p >= BOOT.uncertain_threshold ? "ok" : "uncertain", top, top_label: LABELS[top], top_prob: p };
        }
        const h = (history[m.key] = history[m.key] || []);
        if (r && phase === "running") h.push(r.status === "ok" ? (BOOT.show_prediction ? r.top : "ok") : r.status);
        if (h.length > 60) h.shift();
        return {
          key: m.key, label: m.label, dataset: m.dataset, family: m.family, has_contempt: m.family !== "fer", has_va: !!m.va,
          installed: !m.missing, loaded: on, loading: false, enabled: on, primary: m.key === primaryKey,
          reason: m.missing ? "quraşdırılmayıb — scripts/download_models.py --extra" : on ? null : "İstifadəçi söndürüb",
          ms: on ? m.ms * (0.9 + Math.random() * 0.2) : null, result: r, instant, history: on ? h.slice() : [],
        };
      });
      const active = models.filter((m) => m.enabled);
      const tops = active.filter((m) => m.result && m.result.status === "ok").map((m) => m.result.top);
      const best = tops.sort((a, b) => tops.filter((x) => x === b).length - tops.filter((x) => x === a).length)[0];
      const primary = models.find((m) => m.primary);
      return Promise.resolve({
        phase, camera_active: phase === "running", error: phase === "error" ? "Kamera açılmadı (index 0)" : null,
        frames, records: frames + 1200, self_reports: 3, csv_path: "C:\\ProgramData\\Projects\\faceemotionstest\\logs\\2026-10-22_OP-7F3A.csv",
        next_prompt_s: 372 - frames, prompt_open: !!promptId, preview_enabled: preview,
        face: noface ? { status: "occluded", label: "Üz örtülüb (əl, mikrofon…)", ok: false } : { status: "ok", label: "Üz aşkarlandı", ok: true },
        result: phase === "running" ? primary.result : null,
        instant: phase === "running" ? primary.instant : null,
        latency_ms: 20 + active.reduce((s, m) => s + (m.ms || 0), 0), cpu: 11 + Math.random() * 4,
        compare_mode: compareMode, primary: primaryKey, models, test: phase === "running" ? testState() : null,
        budget: { limit_ms: compareMode === "auto" ? 333.3 : null, used_ms: active.reduce((s, m) => s + (m.ms || 0), 0), interval_ms: 667 },
        cameras: CAMS, camera: CAMS[camIndex] || null, attire: attireState(),
        agreement: BOOT.show_prediction && active.length > 1 ? { agree: best ? tops.filter((x) => x === best).length : 0, total: active.length, top: best || null, top_label: best ? LABELS[best] : null } : null,
      });
    },
    preview() {
      const c = document.createElement("canvas");
      c.width = 320; c.height = 180;
      const g = c.getContext("2d");
      const grd = g.createLinearGradient(0, 0, 320, 180);
      grd.addColorStop(0, "#6b7a99"); grd.addColorStop(1, "#c9b8a8");
      g.fillStyle = grd; g.fillRect(0, 0, 320, 180);
      g.strokeStyle = "#30d158"; g.lineWidth = 2; g.strokeRect(122, 40, 76, 96);
      return Promise.resolve(c.toDataURL("image/jpeg"));
    },
    set_preview(v) { preview = v; return Promise.resolve(true); },
    cameras() { return Promise.resolve({ cameras: CAMS, camera: CAMS[camIndex] }); },
    select_camera(i) {
      if (q.get("camfail") === "1") return Promise.resolve({ ok: false, error: `Kamera açılmadı: ${CAMS[i].name} — başqa proqram istifadə edə bilər` });
      return new Promise((r) => setTimeout(() => { camIndex = i; r({ ok: true, error: null }); }, phase === "running" ? 900 : 0));
    },
    accept_consent() {
      phase = "loading";
      setTimeout(() => { phase = "running"; attireStart = performance.now(); }, 1400);
      return Promise.resolve(true);
    },
    decline() { phase = "stopped"; return Promise.resolve(true); },
    stop() { phase = "stopped"; return Promise.resolve(true); },
    restart() { phase = "consent"; frames = 0; return Promise.resolve(true); },
    close_app() { return Promise.resolve(true); },
    minimize() { return Promise.resolve(true); },
    toggle_compact() { document.body.classList.toggle("compact"); return Promise.resolve(document.body.classList.contains("compact")); },
    demo_prompt() { promptId += 1; promptShown = performance.now(); return Promise.resolve(true); },
    focus_main() { return Promise.resolve(true); },
    prompt_state() {
      if (!promptId) return Promise.resolve(null);
      const left = Math.max(0, 30 - (performance.now() - promptShown) / 1000);
      if (left <= 0) { promptId = 0; return Promise.resolve(null); }
      return Promise.resolve({ id: promptId, seconds_left: left, timeout: 30 });
    },
    answer(id, code) { setTimeout(() => (promptId = 0), 900); return Promise.resolve(true); },
    set_primary(key) { primaryKey = key; enabled[key] = true; return Promise.resolve(true); },
    set_model_enabled(key, on) { enabled[key] = on; if (on && compareMode === "off") compareMode = "on"; return Promise.resolve(true); },
    set_compare_mode(mode) { compareMode = mode; return Promise.resolve(true); },
    island_state() {
      return this.state().then((s) => ({ active: s.camera_active, source: "camera", label: s.result && s.result.top_label, status: s.result && s.result.status }));
    },
  };
})();
