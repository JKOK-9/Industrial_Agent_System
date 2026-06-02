import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const devHost = env.VITE_DEV_HOST || "127.0.0.1";
  const devPort = Number(env.VITE_DEV_PORT || 5173);
  const apiProxyTarget = env.VITE_DEV_API_PROXY_TARGET || "http://127.0.0.1:5000";

  return {
    base: env.VITE_PUBLIC_BASE || "./",
    plugins: [vue()],
    server: {
      host: devHost,
      port: devPort,
      cors: true,
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
      proxy: {
        "/api": apiProxyTarget,
      },
    },
    preview: {
      host: devHost,
      port: Number(env.VITE_PREVIEW_PORT || 4173),
      cors: true,
      headers: {
        "Access-Control-Allow-Origin": "*",
      },
    },
  };
});
