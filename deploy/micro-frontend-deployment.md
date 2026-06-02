# Micro-Frontend Production Deployment

This project has been adapted for a Vue 3 + Vite child application loaded by a micro-frontend host such as micro-app.

## Frontend

1. Build with a relative asset base by default:

   ```powershell
   cd frontend
   npm install
   .\node_modules\.bin\vite.cmd build
   ```

2. Edit `frontend/dist/app-config.js` on the server after build:

   ```js
   window.__APP_CONFIG__ = {
     apiBaseUrl: "https://agent-system.example.com",
     microAppName: "industrial-agent-system",
     baseRoute: "/agent-system"
   };
   ```

   Keep `apiBaseUrl` empty when Nginx proxies `/api` from the same origin.

3. The Vue entry registers `window.unmount`, so the host can unload the child app cleanly.

## Backend

1. Copy `deploy/production.env.example` and adjust CORS/runtime values.
2. Start the Flask backend with these environment variables loaded.
3. The backend exposes APIs under `/api/*`; CORS origins are controlled by `AGENT_SYSTEM_CORS_ORIGINS`.

## Nginx

Use `deploy/nginx.micro-frontend.conf` as a template. It provides:

- static frontend serving with SPA fallback;
- cross-origin headers for child app assets;
- `/api/` reverse proxy to Flask;
- long timeouts for model inference requests.
