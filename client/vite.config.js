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
      "/api": "http://localhost:5000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.js"],
    globals: true,
  },
});
