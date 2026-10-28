import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the Python API runs on :8000 (`physicslens-ui`); in production FastAPI serves dist/ itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
