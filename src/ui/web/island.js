/* "Dynamic Island" tipli kamera indikatoru — kamera açıq olduğu müddətdə həmişə görünür */
"use strict";

mountIcons();
setText($("#label"), BOOT.source === "video" ? "Video oxunur" : "Kamera aktiv");

$("#stop").addEventListener("click", () => Api.call("stop"));
$("#info").addEventListener("click", () => Api.call("focus_main"));

async function poll() {
  const st = await Api.call("island_state");
  if (st) {
    const text = st.label ? `· ${st.label}` : "";
    setText($("#what"), text);
  }
  setTimeout(poll, 1000);
}
poll();
