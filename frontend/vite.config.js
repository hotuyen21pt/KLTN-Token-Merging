import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend target for the dev proxy. Override with BACKEND_URL (e.g. inside Docker:
// "http://backend:5000"). Defaults to the local backend on port 5000.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:5000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    allowedHosts: true,

    proxy: {
      "/predict": BACKEND_URL,
      "/batch_predict": BACKEND_URL,
      "/aspect-stats": BACKEND_URL,
    },
  },
});