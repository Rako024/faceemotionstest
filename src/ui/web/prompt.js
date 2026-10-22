/* Özünüqiymətləndirmə bildirişi: fokus oğurlamır, 30 saniyədə özü bağlanır */
"use strict";

const CATEGORY_COLORS = {
  neutral: "var(--teal)",
  happy: "var(--orange)",
  tired: "var(--indigo)",
  angry: "var(--red)",
  sad: "var(--blue)",
  stressed: "var(--purple)",
};

let current = null; // aktiv sorğunun id-si
let answered = false;

function build() {
  $("#options").innerHTML = BOOT.categories
    .map(
      (c) => `<button class="opt" data-code="${c.code}" style="--c:${CATEGORY_COLORS[c.code] || "var(--gray)"}">
        <span class="swatch"></span><span>${c.label}</span></button>`
    )
    .join("");
  setText($("#decline"), BOOT.decline_label);
}

async function choose(code, el) {
  if (current == null || answered) return;
  answered = true;
  if (el) el.classList.add("chosen");
  const ok = await Api.call("answer", current, code);
  if (ok) {
    setTimeout(() => {
      $("#notif").classList.add("done");
      $("#thanks").classList.add("show");
    }, 180);
  } else {
    answered = false;
  }
}

function reset(id) {
  current = id;
  answered = false;
  $$(".opt").forEach((o) => o.classList.remove("chosen"));
  $("#thanks").classList.remove("show");
  const n = $("#notif");
  n.classList.remove("done", "enter");
  void n.offsetWidth; // animasiyanı yenidən başlat
  n.classList.add("enter");
}

function renderCountdown(ps) {
  const k = Math.max(0, Math.min(1, ps.seconds_left / ps.timeout));
  const bar = $("#bar");
  bar.style.transition = "transform 0.3s linear";
  bar.style.transform = `scaleX(${k})`;
  setText($("#secs"), `${Math.ceil(ps.seconds_left)} san`);
}

async function poll() {
  const ps = await Api.call("prompt_state");
  if (ps) {
    if (ps.id !== current) reset(ps.id);
    if (!answered) renderCountdown(ps);
  } else if (current != null && !answered) {
    current = null; // vaxt bitdi — Python pəncərəni gizlədir
  }
  setTimeout(poll, 250);
}

/* Python pəncərəni göstərməzdən əvvəl çağırır — köhnə vəziyyət bir an belə görünməsin */
window.__open = (id, secondsLeft, timeout) => {
  reset(id);
  renderCountdown({ seconds_left: secondsLeft, timeout });
};

$("#options").addEventListener("click", (e) => {
  const opt = e.target.closest(".opt");
  if (opt) choose(opt.dataset.code, opt);
});
$("#decline").addEventListener("click", () => choose("declined"));
$("#close").addEventListener("click", () => choose("declined"));

mountIcons();
build();
poll();
