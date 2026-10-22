/* Əsas pəncərə: razılıq → yüklənmə → geniş panel (kamera + bütün modellər) → son ekran */
"use strict";

const CIRC = 2 * Math.PI * 56;
const TIMELINE_LEN = 60;
const state = { phase: null, screen: null, compact: false, previewTimer: null, pollTimer: null, last: null, cardKeys: "", otherKeys: "" };

const EMO_KEYS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"];

function bindBoot() {
  const b = BOOT;
  const values = {
    operator_id: b.operator_id,
    model: b.model,
    engine: b.engine_label,
    compare: b.compare_label,
    fps: b.analysis_fps,
    window: b.window_seconds,
    source: b.source === "video" ? "Video fayl" : "Veb-kamera",
    interval: b.prompt_interval_min,
    timeout: b.prompt_timeout_s,
  };
  $$("[data-bind]").forEach((el) => setText(el, values[el.dataset.bind] ?? ""));
  if (!b.prompts_enabled) $$('[data-show="prompts"]').forEach((el) => el.remove());
  if (!b.research_enabled) {
    $$('[data-show="research"]').forEach((el) => el.remove());
    $("#setup-consent").remove();
    $("#setup-note").remove();
  }
  setText($("#setup-sub"), `${b.test_expressions} ifadə × ${b.test_rounds} raund · təxminən ${b.test_minutes} dəqiqə`);
  if (!b.prompts_enabled) $("#m-next-card").remove();
  setText($("#live-label"), b.source === "video" ? "Video oxunur" : "Kamera aktiv");
  if (b.source === "video") $$('[data-show="camera"]').forEach((el) => el.remove());
  buildLegend();
}

/* ---------- ekran keçidləri ---------- */
/* Keçid animasiyası bitəndən sonra qeyri-aktiv ekran display:none olur — Chromium backdrop-filter-li
   şüşə kartları opacity:0 / visibility:hidden valideyndə də çəkir və aktiv ekranın arxasında görünür. */
function showScreen(id) {
  $$(".screen").forEach((s) => {
    clearTimeout(s._hideTimer);
    if (s.id === id) {
      s.style.display = "";
      requestAnimationFrame(() => s.classList.add("active"));
    } else if (s.style.display !== "none") {
      s.classList.remove("active");
      s._hideTimer = setTimeout(() => {
        if (!s.classList.contains("active")) s.style.display = "none";
      }, 520);
    }
  });
  $("#toolbar").hidden = id !== "running";
}

function screenForPhase(phase, st) {
  if (phase === "running" && st && st.test) return "test";
  return { consent: "consent", loading: "loading", running: "running" }[phase] || "ended";
}

function labelOf(key) {
  const c = BOOT.classes.find((x) => x.key === key);
  return c ? c.label : key;
}

/* ---------- əsas model (sol sütun) ---------- */
function renderHero(st) {
  const hero = $("#hero");
  const ring = $("#ring-value");
  const r = st.result;
  let mode;
  if (!r || r.status === "no_face") mode = "no-face";
  else if (r.status === "warming_up") mode = "warming";
  else if (!BOOT.show_prediction) mode = "blind";
  else mode = r.status; // ok | uncertain

  hero.className = "glass hero " + mode;
  hero.classList.toggle("indeterminate", mode === "warming");
  hero.classList.toggle("show-icon", mode === "no-face" || mode === "blind");
  const primary = (st.models || []).find((m) => m.primary);
  setText($("#hero-model"), primary ? `Əsas model · ${primary.label}` : "");

  const kicker = `Son ${BOOT.window_seconds} saniyə`;
  renderInstant($("#hero-instant"), mode === "no-face" ? null : st.instant, st.frames);
  if (mode === "no-face") {
    $("#ring-icon").innerHTML = icon("face-off");
    ring.style.strokeDashoffset = CIRC;
    setText($("#hero-kicker"), "Təxmin edilmir");
    setText($("#hero-label"), "Üz yoxdur");
    setText($("#hero-sub"), st.face ? st.face.label : "Kadr gözlənilir");
    setAmbient(null);
  } else if (mode === "warming") {
    ring.style.removeProperty("stroke-dashoffset");
    setText($("#hero-kicker"), kicker);
    setText($("#hero-label"), "Toplanır…");
    setText($("#hero-sub"), `${r ? r.n_frames : 0} / ${BOOT.min_frames} kadr toplanır`);
  } else if (mode === "blind") {
    $("#ring-icon").innerHTML = icon("check");
    hero.style.setProperty("--ring", "var(--green)");
    ring.style.strokeDashoffset = 0;
    setText($("#hero-kicker"), kicker);
    setText($("#hero-label"), "Ölçülür");
    setText($("#hero-sub"), "Nəticə gizlidir (kor rejim)");
  } else {
    hero.style.setProperty("--ring", mode === "ok" ? `var(--emo-${r.top})` : "var(--gray)");
    ring.style.strokeDashoffset = CIRC * (1 - r.top_prob);
    tweenNumber($("#ring-pct"), r.top_prob * 100, (v) => `${Math.round(v)}%`);
    setText($("#hero-kicker"), kicker);
    if (mode === "ok") {
      setText($("#hero-label"), r.top_label);
      setText($("#hero-sub"), `${r.n_frames} kadrın ortalaması`);
      setAmbient(r.top);
    } else {
      setText($("#hero-label"), "Qeyri-müəyyən");
      setText($("#hero-sub"), `Ən yüksək: ${r.top_label} ${pct(r.top_prob)} — ${Math.round(BOOT.uncertain_threshold * 100)}%-dən az`);
      setAmbient(null);
    }
  }
}

/* Anlıq: son kadrın öz nəticəsi (ortalamasız). Nöqtə hər yeni analiz kadrında bir dəfə döyünür. */
function renderInstant(el, inst, frames) {
  if (!el) return;
  el.hidden = !inst || !BOOT.show_prediction || !(inst.top || inst.status === "no_face");
  if (el.hidden) return;
  const ok = inst.status === "ok";
  if (inst.top) {
    setText(el.querySelector(".i-label"), inst.top_label);
    setText(el.querySelector(".i-pct"), pct(inst.top_prob));
  } else {
    setText(el.querySelector(".i-label"), "üz yoxdur");
    setText(el.querySelector(".i-pct"), "");
  }
  el.classList.toggle("weak", !ok);
  el.style.setProperty("--dot", ok ? `var(--emo-${inst.top})` : "var(--label-3)");
  el.setAttribute("aria-label", `Anlıq: ${inst.top ? `${inst.top_label} ${pct(inst.top_prob)}${ok ? "" : ", qeyri-müəyyən"}` : "üz yoxdur"}`);
  if (frames != null && el.dataset.frame !== String(frames)) {
    el.dataset.frame = String(frames);
    const dot = el.querySelector(".dot");
    dot.classList.remove("beat");
    void dot.offsetWidth; // animasiyanı yenidən başlat
    dot.classList.add("beat");
  }
}

function renderMetrics(st) {
  const face = $("#m-face");
  if (st.face) {
    setText(face, st.face.label);
    face.className = "metric small " + (st.face.ok ? "ok" : "warn");
    setText($("#preview-badge-text"), st.face.label);
    $("#preview-badge").classList.toggle("ok", st.face.ok);
  }
  setText($("#m-latency"), st.latency_ms != null ? `${Math.round(st.latency_ms)} ms` : "—");
  setText($("#m-cpu"), st.cpu != null ? `${Math.round(st.cpu)}%` : "—");
  const next = $("#m-next");
  if (next) setText(next, st.prompt_open ? "İndi" : mmss(st.next_prompt_s));
  setText($("#footer-stats"), `${fmtInt(st.records)} sətir yazıldı · ${st.self_reports} sorğu cavabı`);
}

/* ---------- modellər (sağ sütun) ---------- */
function cardSkeleton(m) {
  const bars = EMO_KEYS.map(
    (k) => `<span class="nm" data-k="${k}">${labelOf(k)}</span><span class="br" style="--c: var(--emo-${k})"><i data-k="${k}"></i></span><span class="vl" data-k="${k}">—</span>`
  ).join("");
  return `<article class="glass mcard" data-key="${m.key}">
    <div class="mc-head">
      <span class="mc-name"></span><span class="mc-ds"></span>
      <div class="mc-actions">
        <button class="mc-star" data-act="primary" aria-label="Əsas model et" title="Əsas model et">${icon("star")}</button>
        <button class="switch sm" role="switch" data-act="toggle" aria-label="Modeli aktiv et"></button>
      </div>
    </div>
    <div class="mc-body">
      <div class="mc-top"><span class="mc-label"></span><span class="mc-pct num"></span></div>
      <p class="instant sm" hidden><span class="dot"></span><span class="i-cap">Anlıq</span><span class="i-label"></span><span class="i-pct num"></span></p>
      <div class="mc-bars">${bars}</div>
      <div class="mc-va" hidden></div>
    </div>
    <p class="mc-reason" hidden></p>
    <div class="mc-loading" hidden><span class="spinner sm"></span>Yüklənir…</div>
    <div class="mc-foot"><span class="mc-ms num"></span><span class="mc-fam"></span></div>
  </article>`;
}

function mountSpinner(el) {
  el.innerHTML = Array.from(
    { length: 12 },
    (_, i) => `<i style="transform: rotate(${i * 30}deg); animation-delay: ${(i - 12) / 12}s"></i>`
  ).join("");
}

function updateCard(card, m) {
  card.classList.toggle("primary", m.primary);
  card.classList.toggle("off", !m.enabled);
  setText(card.querySelector(".mc-name"), m.label);
  setText(card.querySelector(".mc-ds"), m.dataset);
  const star = card.querySelector(".mc-star");
  star.classList.toggle("on", m.primary);
  star.disabled = !m.installed;
  const sw = card.querySelector('[data-act="toggle"]');
  sw.setAttribute("aria-checked", String(m.enabled));
  sw.disabled = m.primary || !m.installed;
  sw.title = m.primary ? "Əsas model həmişə işləyir" : m.enabled ? "Söndür" : "Aktiv et";

  const loading = card.querySelector(".mc-loading");
  loading.hidden = !m.loading;
  if (m.loading && !loading.querySelector(".spinner i")) mountSpinner(loading.querySelector(".spinner"));
  const reason = card.querySelector(".mc-reason");
  reason.hidden = m.enabled || m.loading || !m.reason;
  setText(reason, m.reason || "");

  const r = m.result;
  const label = card.querySelector(".mc-label");
  const pctEl = card.querySelector(".mc-pct");
  if (!m.enabled) {
    setText(label, m.installed ? "Söndürülüb" : "Quraşdırılmayıb");
    label.className = "mc-label muted";
    setText(pctEl, "");
  } else if (!r || r.status === "no_face") {
    setText(label, "Üz yoxdur");
    label.className = "mc-label muted";
    setText(pctEl, "");
  } else if (r.status === "warming_up") {
    setText(label, "Toplanır…");
    label.className = "mc-label muted";
    setText(pctEl, `${r.n_frames}/${BOOT.min_frames}`);
  } else if (!r.top) {
    setText(label, "Ölçülür");
    label.className = "mc-label muted";
    setText(pctEl, "");
  } else if (r.status === "uncertain") {
    setText(label, "Qeyri-müəyyən");
    label.className = "mc-label muted";
    setText(pctEl, `${r.top_label} ${pct(r.top_prob)}`);
  } else {
    setText(label, r.top_label);
    label.className = "mc-label";
    setText(pctEl, pct(r.top_prob));
  }

  const inst = m.enabled && r && r.status !== "no_face" ? m.instant : null;
  renderInstant(card.querySelector(".instant"), inst, state.last && state.last.frames);

  const probs = r && r.probs ? Object.fromEntries(r.probs.map((p) => [p.key, p.p])) : null;
  EMO_KEYS.forEach((k) => {
    const v = probs ? probs[k] : undefined;
    const unsupported = k === "Contempt" && !m.has_contempt;
    card.querySelector(`.br i[data-k="${k}"]`).style.width = v != null ? `${(v * 100).toFixed(1)}%` : "0%";
    const vl = card.querySelector(`.vl[data-k="${k}"]`);
    setText(vl, unsupported ? "yox" : v != null ? pct(v) : "—");
    vl.classList.toggle("na", unsupported);
    vl.title = unsupported ? "Bu modeldə “Həqarət” sinfi yoxdur (FER2013)" : "";
    card.querySelector(`.nm[data-k="${k}"]`).classList.toggle("top", !!(r && r.top === k && r.status === "ok"));
  });

  const va = card.querySelector(".mc-va");
  va.hidden = !(r && r.valence != null);
  if (!va.hidden) setText(va, `Valentlik ${r.valence >= 0 ? "+" : ""}${r.valence.toFixed(2)} · Oyanma ${r.arousal >= 0 ? "+" : ""}${r.arousal.toFixed(2)}`);
  setText(card.querySelector(".mc-ms"), m.ms != null ? `${m.ms < 10 ? m.ms.toFixed(1) : Math.round(m.ms)} ms` : "—");
  setText(card.querySelector(".mc-fam"), m.family === "fer" ? "7 sinif" : m.has_va ? "8 sinif + VA" : "8 sinif");
}

function otherRow(m) {
  return `<div class="orow" data-key="${m.key}">
    <span class="mc-name"></span><span class="mc-ds"></span><span class="o-reason"></span>
    <div class="mc-actions">
      <button class="mc-star" data-act="primary" aria-label="Əsas model et" title="Əsas model et">${icon("star")}</button>
      <button class="switch sm" role="switch" data-act="toggle" aria-label="Modeli aktiv et"></button>
    </div></div>`;
}

function updateOther(row, m) {
  setText(row.querySelector(".mc-name"), m.label);
  setText(row.querySelector(".mc-ds"), m.dataset);
  setText(row.querySelector(".o-reason"), m.loading ? "Yüklənir…" : m.reason || "Söndürülüb");
  row.querySelector(".mc-star").disabled = !m.installed;
  const sw = row.querySelector('[data-act="toggle"]');
  sw.setAttribute("aria-checked", String(m.loading));
  sw.disabled = !m.installed || m.loading;
}

/* Aktiv modellər — tam kart (8 ehtimal); qalanları zaman xəttindən sonra kompakt siyahıda */
function renderCards(st) {
  const models = st.models || [];
  const active = models.filter((m) => m.enabled);
  const others = models.filter((m) => !m.enabled);

  const box = $("#cards");
  const keys = active.map((m) => m.key).join("|");
  if (keys !== state.cardKeys) {
    box.innerHTML = active.map(cardSkeleton).join("");
    state.cardKeys = keys;
  }
  active.forEach((m) => updateCard(box.querySelector(`[data-key="${m.key}"]`), m));

  const list = $("#others");
  const okeys = others.map((m) => m.key).join("|");
  if (okeys !== state.otherKeys) {
    list.innerHTML = others.map(otherRow).join("");
    state.otherKeys = okeys;
  }
  $("#others-wrap").hidden = !others.length;
  others.forEach((m) => updateOther(list.querySelector(`[data-key="${m.key}"]`), m));
}

function renderTimeline(st) {
  const box = $("#timeline");
  const models = (st.models || []).filter((m) => m.enabled);
  if (!models.length) {
    box.innerHTML = '<p class="empty-note">Aktiv model yoxdur</p>';
    return;
  }
  box.innerHTML = models
    .map((m) => {
      const h = m.history.slice(-TIMELINE_LEN);
      const pad = TIMELINE_LEN - h.length;
      const cells = Array.from({ length: pad }, () => "<i></i>").concat(
        h.map((c) => {
          if (EMO_KEYS.includes(c)) return `<i style="background: var(--emo-${c})" title="${labelOf(c)}"></i>`;
          return `<i class="${c}"></i>`;
        })
      );
      return `<span class="tl-name${m.primary ? " primary" : ""}">${m.label}</span><div class="tl-cells">${cells.join("")}</div>`;
    })
    .join("");
}

function buildLegend() {
  const items = BOOT.show_prediction
    ? BOOT.classes.map((c) => `<li><span class="k" style="background: var(--emo-${c.key})"></span>${c.label}</li>`)
    : ['<li><span class="k" style="background: var(--green)"></span>Ölçülür</li>'];
  items.push('<li><span class="k" style="background: color-mix(in srgb, var(--gray) 45%, transparent)"></span>Qeyri-müəyyən</li>');
  items.push('<li><span class="k" style="background: color-mix(in srgb, var(--label) 22%, transparent)"></span>Üz yoxdur</li>');
  $("#tl-legend").innerHTML = items.join("");
}

function renderModelsHead(st) {
  const chip = $("#agree-chip");
  const a = st.agreement;
  chip.hidden = !a || a.total < 2;
  if (a && a.total >= 2) {
    setText(chip, a.top ? `Razılaşma ${a.agree} / ${a.total} · ${a.top_label}` : `Razılaşma 0 / ${a.total}`);
    chip.className = "chip " + (a.agree === a.total ? "agree-all" : "agree-some");
  }
  const b = st.budget;
  const fill = $("#budget-fill");
  if (b) {
    const limit = b.limit_ms || b.interval_ms;
    const k = limit ? b.used_ms / limit : 0;
    fill.style.width = `${Math.min(100, k * 100).toFixed(1)}%`;
    fill.className = k > 1 ? "over" : k > 0.75 ? "warn" : "";
    setText(
      $("#budget-label"),
      b.limit_ms ? `Hesablama ${Math.round(b.used_ms)} / ${Math.round(b.limit_ms)} ms` : `Hesablama ${Math.round(b.used_ms)} ms · interval ${b.interval_ms} ms`
    );
  }
}

function renderToolbar(st) {
  const primary = (st.models || []).find((m) => m.primary);
  setText($("#primary-label"), primary ? primary.label : "—");
  $$("#compare-seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === st.compare_mode)));
  if (!$("#primary-menu").hidden) buildMenu(st);
}

function buildMenu(st) {
  const menu = $("#primary-menu");
  const models = st.models || [];
  const rows = (fam) =>
    models
      .filter((m) => m.family === fam)
      .map(
        (m) => `<button role="menuitemradio" aria-checked="${m.primary}" data-key="${m.key}" ${m.installed ? "" : "disabled"}>
          <span class="check">${m.primary ? icon("check") : ""}</span>
          <span>${m.label}<br><span class="m-sub">${m.installed ? m.dataset + (m.ms != null ? ` · ${m.ms < 10 ? m.ms.toFixed(1) : Math.round(m.ms)} ms` : "") : "quraşdırılmayıb"}</span></span>
          <span class="m-sub">${m.enabled ? "aktiv" : ""}</span></button>`
      )
      .join("");
  menu.innerHTML = rows("emotiefflib") + '<div class="m-sep"></div>' + rows("fer");
}

/* ---------- son ekran ---------- */
function renderEnded(st) {
  const iconEl = $("#ended-icon");
  const kind = st.phase;
  iconEl.className = "ended-icon" + (kind === "error" ? " error" : kind === "finished" ? " done" : "");
  iconEl.innerHTML = icon(kind === "error" ? "alert" : kind === "finished" ? "check" : "video-off");
  setText($("#ended-title"), { stopped: "Dayandırıldı", finished: "Video bitdi", error: "Başlamaq alınmadı" }[kind] || "");
  setText(
    $("#ended-sub"),
    kind === "error" ? st.error || "Naməlum xəta" : kind === "finished" ? "Bütün kadrlar emal olundu. Nəticələr saxlanıldı." : "Kamera bağlıdır. Nəticələr saxlanıldı."
  );
  $("#ended-card").hidden = kind === "error" && !st.records;
  setText($("#ended-records"), fmtInt(st.records));
  setText($("#ended-reports"), st.self_reports);
  const path = $("#ended-path");
  setText(path, st.csv_path ? st.csv_path.split(/[\\/]/).slice(-2).join("/") : "—");
  path.title = st.csv_path || "";
  setText($("#restart-label"), kind === "error" ? "Yenidən cəhd et" : "Yenidən başla");
}

/* ---------- nəzarətli test ---------- */
const PHASE_LABELS = { intro: "Hazırlaşın", prepare: "Hazırlaşın", hold: "İndi saxlayın", rest: "Rahatlayın" };

function renderTest(st) {
  const t = st.test;
  $("#test-run").hidden = t.done;
  $("#test-result").hidden = !t.done;
  if (t.done) return renderTestResult(t);

  setText($("#test-progress"), t.phase === "intro" ? `${t.total} addım · ${t.rounds} raund` : `Raund ${t.round} / ${t.rounds} · Addım ${t.step + 1} / ${t.total}`);
  const sig = t.steps.join("|");
  const dots = $("#test-dots");
  if (dots.dataset.sig !== sig) {
    dots.innerHTML = t.steps.map((e) => `<i style="--c: var(--emo-${e})" title="${labelOf(e)}"></i>`).join("");
    dots.dataset.sig = sig;
  }
  [...dots.children].forEach((d, i) => {
    d.className = i < t.step || (i === t.step && t.phase === "rest") ? "done" : i === t.step && t.phase !== "intro" ? "now" : "";
  });

  const pill = $("#test-phase");
  setText(pill, PHASE_LABELS[t.phase] || "");
  pill.className = "phase-pill " + t.phase;

  const emo = $("#test-emotion");
  let color;
  if (t.phase === "intro") {
    emo.innerHTML = "Başlayırıq";
    setText($("#test-hint"), "Hər ifadəni göstərişə uyğun saxlayın. İlk ifadə: " + t.label);
    color = "var(--accent)";
  } else if (t.phase === "rest") {
    const next = t.steps[t.step + 1];
    emo.innerHTML = "Rahatlayın";
    setText($("#test-hint"), next ? `Növbəti: ${labelOf(next)}` : "Son addım bitdi");
    color = "var(--gray)";
  } else {
    emo.innerHTML = `<span class="sw" style="--c: var(--emo-${t.expected})"></span>${t.label}`;
    setText($("#test-hint"), t.hint);
    color = `var(--emo-${t.expected})`;
  }
  const ring = $("#test-ring");
  ring.style.stroke = color; // SVG-də var() keçidi gecikir — rəng birbaşa verilir
  const k = t.phase_seconds ? t.seconds_left / t.phase_seconds : 0;
  ring.style.strokeDashoffset = CIRC * (1 - k);
  setText($("#test-secs"), Math.ceil(t.seconds_left));
  if (st.face) {
    setText($("#test-badge-text"), st.face.label);
    $("#test-badge").classList.toggle("ok", st.face.ok);
  }
}

function renderTestResult(t) {
  setText($("#result-title"), t.cancelled ? "Test dayandırıldı — qismən nəticə" : "Test nəticəsi");
  const sub = $("#result-sub");
  const base = `Tester ${t.tester_id || "—"} · hər kadrın öz nəticəsi · yalnız üz görünən kadrlar`;
  sub.innerHTML = t.saved_frames != null
    ? `${base} · <span class="result-saved">${t.saved_frames} şəkil saxlanıldı (dataset/${t.dataset_session})</span>`
    : base;
  const rows = t.summary || [];
  // "Ən yaxşı" yalnız aydın üstünlük olanda (bərabərlikdə və ya 0%-də göstərilmir)
  const lead = rows.length > 1 && rows[0].balanced > 0 && rows[0].balanced > (rows[1].balanced ?? -1);
  const best = lead ? rows[0].key : null;
  const fmt = (v) => (v == null ? "—" : pct(v));
  $("#result-table").innerHTML =
    `<div class="rt-row head"><span>Model</span><span>Dataset</span><span>Düzgünlük (bütün kadrlar)</span><span class="r">Balanslı</span><span class="r">Qeyri-müəyyən</span><span class="r">Kadr</span></div>` +
    rows
      .map(
        (r) => `<div class="rt-row">
          <span class="nm">${r.label}${r.key === best ? '<span class="best">ƏN YAXŞI</span>' : ""}<small>${r.key}</small></span>
          <span class="r">${r.dataset}</span>
          <span class="acc"><span class="num">${fmt(r.accuracy)}</span><span class="track"><i style="width:${((r.accuracy || 0) * 100).toFixed(1)}%"></i></span></span>
          <span class="r">${fmt(r.balanced)}</span><span class="r">${fmt(r.uncertain)}</span><span class="r">${r.face} / ${r.frames}</span></div>`
      )
      .join("");
  const ex = t.expressions || [];
  const heat = $("#result-heat");
  heat.innerHTML =
    `<div class="heat" style="grid-template-columns: minmax(120px, auto) repeat(${ex.length}, minmax(70px, 1fr))">` +
    `<span></span>` +
    ex.map((e) => `<span class="h"><span class="sw" style="--c: var(--emo-${e.key})"></span>${e.label}</span>`).join("") +
    rows
      .map(
        (r) =>
          `<span class="m">${r.label}</span>` +
          ex
            .map((e) => {
              const v = r.per_expression[e.key];
              if (v == null) return `<span class="c na" title="${e.label}: kadr yoxdur">—</span>`;
              return `<span class="c${v > 0.6 ? " hi" : ""}" style="--v: ${v.toFixed(3)}" title="${r.label} · ${e.label}: ${pct(v)}">${pct(v)}</span>`;
            })
            .join("")
      )
      .join("") +
    `</div>`;
}

function render(st) {
  if (!st) return;
  const prev = state.last;
  state.last = st;
  renderCamera(st);
  const screen = screenForPhase(st.phase, st);
  if (st.phase !== state.phase || screen !== state.screen) {
    state.phase = st.phase;
    state.screen = screen;
    showScreen(screen);
    if (st.phase !== "running") setAmbient(null);
    syncPreview();
  } else if (prev && prev.preview_enabled !== st.preview_enabled) {
    syncPreview();
  }
  if (screen === "test") {
    renderTest(st);
    return;
  }
  if (st.phase === "running") {
    renderHero(st);
    renderMetrics(st);
    renderModelsHead(st);
    renderCards(st);
    renderTimeline(st);
    renderToolbar(st);
    $("#preview-switch").setAttribute("aria-checked", String(st.preview_enabled));
    $("#preview-collapse").classList.toggle("open", st.preview_enabled);
  } else if (screenForPhase(st.phase) === "ended") {
    renderEnded(st);
  }
}

async function poll() {
  clearTimeout(state.pollTimer);
  render(await Api.call("state"));
  const fast = state.screen === "test" && state.last && state.last.test && state.last.test.active;
  state.pollTimer = setTimeout(poll, fast ? 250 : state.phase === "running" || state.phase === "loading" ? 500 : 1000);
}

/* ---------- önizləmə ---------- */
function syncPreview() {
  clearTimeout(state.previewTimer);
  const on = state.last && state.last.phase === "running" && state.last.preview_enabled;
  if (!on) {
    $("#preview-img").removeAttribute("src");
    $("#test-preview").removeAttribute("src");
    return;
  }
  const tick = async () => {
    const url = await Api.call("preview");
    if (url) {
      $("#preview-img").src = url;
      $("#test-preview").src = url;
    }
    if (state.last && state.last.phase === "running" && state.last.preview_enabled) {
      state.previewTimer = setTimeout(tick, 1000 / BOOT.preview_fps);
    }
  };
  tick();
}

/* ---------- hərəkətlər ---------- */
const actions = {
  async accept(btn) {
    btn.disabled = true;
    showScreen("loading");
    await Api.call("accept_consent");
    btn.disabled = false;
    poll();
  },
  decline: () => Api.call("decline"),
  async stop() {
    await Api.call("stop");
    poll();
  },
  async restart() {
    await Api.call("restart");
    poll();
  },
  close: () => Api.call("close_app"),
  minimize: () => Api.call("minimize"),
  "start-test"() {
    openSetup();
  },
  "close-setup"() {
    $("#test-setup").hidden = true;
  },
  async "confirm-test"(btn) {
    const id = $("#tester-id").value.trim();
    const err = $("#setup-error");
    if (!/^[A-Za-z0-9_-]{2,32}$/.test(id) || !/\d/.test(id)) {
      setText(err, "Anonim ID yazın: 2–32 simvol, ən azı bir rəqəm (məs. T-01). Ad yazmayın.");
      err.hidden = false;
      $("#tester-id").focus();
      return;
    }
    const sw = $("#save-switch");
    const save = !!sw && sw.getAttribute("aria-checked") === "true";
    btn.disabled = true;
    const res = await Api.call("start_test", id, save);
    btn.disabled = false;
    if (!res || !res.ok) {
      setText(err, (res && res.error) || "Test başlamadı");
      err.hidden = false;
      return;
    }
    state.lastTester = id;
    $("#test-setup").hidden = true;
    poll();
  },
  async "cancel-test"() {
    await Api.call("cancel_test");
    poll();
  },
  async "dismiss-test"() {
    await Api.call("dismiss_test");
    poll();
  },
  async compact() {
    state.compact = await Api.call("toggle_compact");
    document.body.classList.toggle("compact", !!state.compact);
  },
};

/* Növbəti tester üçün ID təklifi: T-01 → T-02. Razılıq açarı HƏR DƏFƏ söndürülmüş açılır. */
function openSetup() {
  const input = $("#tester-id");
  const last = state.lastTester;
  const m = last && last.match(/^(.*?)(\d+)$/);
  input.value = m ? m[1] + String(Number(m[2]) + 1).padStart(m[2].length, "0") : "T-01";
  const sw = $("#save-switch");
  if (sw) sw.setAttribute("aria-checked", "false");
  $("#setup-error").hidden = true;
  $("#test-setup").hidden = false;
  setTimeout(() => input.select(), 50);
}

function closeMenu() {
  $$(".menu").forEach((m) => (m.hidden = true));
  $$('[aria-haspopup="menu"]').forEach((b) => b.setAttribute("aria-expanded", "false"));
}

/* ---------- kamera seçimi ---------- */
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function renderCamera(st) {
  const cam = st.camera;
  const label = $("#camera-label");
  if (label && !state.cameraBusy) {
    setText(label, cam ? cam.name : "Kamera");
    $("#camera-btn").title = cam ? `Kamera: ${cam.name}` : "Kamera";
  }
  const consent = $("#consent-camera");
  if (consent) {
    setText(consent, cam ? cam.name : st.cameras && st.cameras.length === 0 ? "Kamera tapılmadı — config-dəki index ilə cəhd olunacaq" : "—");
  }
  for (const id of ["#camera-menu", "#consent-camera-menu"]) {
    const menu = $(id);
    if (menu && !menu.hidden) buildCameraMenu(menu, st);
  }
}

function buildCameraMenu(menu, st) {
  const cams = st.cameras || [];
  const cur = st.camera ? st.camera.index : null;
  const live = st.phase === "running";
  const testOn = !!(st.test && st.test.active);
  const items = cams
    .map(
      (c) => `<button role="menuitemradio" aria-checked="${c.index === cur}" data-cam="${c.index}" ${testOn ? "disabled" : ""}>
        <span class="check">${c.index === cur ? icon("check") : ""}</span>
        <span>${esc(c.name)}</span>
        <span class="m-sub">${c.index === cur && live ? "işləyir" : ""}</span></button>`
    )
    .join("");
  const notes = (cams.length ? "" : '<p class="m-note">Kamera tapılmadı</p>') + (testOn ? '<p class="m-note">Test gedərkən kamera dəyişdirilmir</p>' : "");
  menu.innerHTML = `${items}${notes}<div class="m-sep"></div>
    <button role="menuitem" data-cam-refresh><span class="check">${icon("refresh")}</span><span>Siyahını yenilə</span><span></span></button>`;
}

async function refreshCameras(menu) {
  const res = await Api.call("cameras", true); // taxılıb-çıxarılan kameraları da gör
  if (!res || !state.last) return;
  Object.assign(state.last, res);
  renderCamera(state.last);
  if (menu && !menu.hidden) buildCameraMenu(menu, state.last);
}

function wireCameraMenu(btn, menu) {
  if (!btn || !menu) return;
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = menu.hidden;
    closeMenu();
    if (!open) return;
    if (state.last) buildCameraMenu(menu, state.last);
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    refreshCameras(menu);
  });
  menu.addEventListener("click", async (e) => {
    if (e.target.closest("[data-cam-refresh]")) {
      refreshCameras(menu);
      return;
    }
    const item = e.target.closest("button[data-cam]");
    if (!item || item.disabled) return;
    closeMenu();
    const idx = Number(item.dataset.cam);
    const st = state.last;
    if (!st || (st.camera && st.camera.index === idx)) return;
    const cam = (st.cameras || []).find((c) => c.index === idx);
    const live = st.phase === "running";
    if (live) {
      state.cameraBusy = true;
      setText($("#camera-label"), "Açılır…");
      $("#camera-btn").classList.add("busy");
    }
    const res = await Api.call("select_camera", idx);
    state.cameraBusy = false;
    $("#camera-btn").classList.remove("busy");
    if (!res || !res.ok) toast((res && res.error) || "Kamera dəyişdirilmədi", true);
    else if (live && cam) toast(`Kamera: ${cam.name}`);
    poll();
  });
}

function toast(message, error = false) {
  const t = $("#toast");
  setText(t, message);
  t.classList.toggle("error", error);
  t.hidden = false;
  requestAnimationFrame(() => t.classList.add("show"));
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    t.classList.remove("show");
    state.toastTimer = setTimeout(() => (t.hidden = true), 300);
  }, error ? 5000 : 2500);
}

function wire() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (btn && actions[btn.dataset.action]) actions[btn.dataset.action](btn);
    if (!e.target.closest(".menu-wrap")) closeMenu();
  });

  const toggle = $("#details-toggle");
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    $("#details").classList.toggle("open", open);
  });

  $("#preview-switch").addEventListener("click", async (e) => {
    const on = e.currentTarget.getAttribute("aria-checked") !== "true";
    e.currentTarget.setAttribute("aria-checked", String(on));
    $("#preview-collapse").classList.toggle("open", on);
    await Api.call("set_preview", on);
    if (state.last) state.last.preview_enabled = on;
    syncPreview();
    if (on) $(".left .scroll").scrollTo({ top: 0, behavior: "smooth" });
  });

  // Model kartları və siyahı: ulduz → əsas model; açar → aktiv/söndür
  $(".right").addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-act]");
    if (!btn || btn.disabled) return;
    const key = btn.closest("[data-key]").dataset.key;
    if (btn.dataset.act === "primary") await Api.call("set_primary", key);
    else await Api.call("set_model_enabled", key, btn.getAttribute("aria-checked") !== "true");
    poll();
  });

  wireCameraMenu($("#camera-btn"), $("#camera-menu"));
  wireCameraMenu($("#consent-camera-btn"), $("#consent-camera-menu"));
  $("#primary-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    const menu = $("#primary-menu");
    const open = menu.hidden;
    closeMenu();
    if (open && state.last) buildMenu(state.last);
    menu.hidden = !open;
    $("#primary-btn").setAttribute("aria-expanded", String(open));
  });
  $("#primary-menu").addEventListener("click", async (e) => {
    const item = e.target.closest("button[data-key]");
    if (!item || item.disabled) return;
    closeMenu();
    await Api.call("set_primary", item.dataset.key);
    poll();
  });
  $("#compare-seg").addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-mode]");
    if (!b) return;
    $$("#compare-seg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    await Api.call("set_compare_mode", b.dataset.mode);
    poll();
  });

  const saveSwitch = $("#save-switch");
  if (saveSwitch) {
    saveSwitch.addEventListener("click", () => {
      saveSwitch.setAttribute("aria-checked", String(saveSwitch.getAttribute("aria-checked") !== "true"));
    });
  }
  $("#tester-id").addEventListener("input", () => ($("#setup-error").hidden = true));
  $("#tester-id").addEventListener("keydown", (e) => {
    if (e.key === "Enter") actions["confirm-test"]($('[data-action="confirm-test"]'));
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeMenu();
      $("#test-setup").hidden = true;
    }
    // Ctrl+Shift+P — sorğunu dərhal göstər (pilotda nümayiş üçün)
    if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "p") Api.call("demo_prompt");
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) poll();
  });
}

mountIcons();
bindBoot();
mountSpinner($("#loading .spinner"));
wire();
showScreen(screenForPhase(BOOT.initial_phase || "consent"));
poll();
