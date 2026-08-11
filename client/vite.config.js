import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built to client/dist and served as static assets by the same Flask host
// (U15's approach) — app.py points its static folder there rather than this
// project taking on a second server. The dev proxy exists only so `npm run
// dev` can hit the real Flask API on :5000 without a CORS workaround; the
// built app is same-origin in production and never uses it.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // Overridable because port 5000 is not always available to Flask --
      // macOS hands it to the AirPlay receiver by default, so a developer
      // there runs the API on another port and would otherwise have to edit
      // (and avoid committing) this file to use `npm run dev` at all.
      "/api": process.env.API_PROXY_TARGET || "http://localhost:5000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.js"],
    globals: true,
  },
});
