import { createApp } from "vue";
import App from "./App.vue";
import "./styles.css";

let app = null;

export function mount() {
  if (app) return app;
  const target = document.querySelector("#app");
  if (!target) return null;
  app = createApp(App);
  app.mount(target);
  return app;
}

export function unmount() {
  if (!app) return;
  app.unmount();
  app = null;
}

mount();

window.unmount = unmount;
window.addEventListener("unmount", unmount);
