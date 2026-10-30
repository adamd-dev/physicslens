import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The public landing page: its own entry and public files (robots, sitemap, og image), no API.
export default defineConfig({
  plugins: [react()],
  publicDir: "landing-public",
  build: {
    outDir: "dist-landing",
    emptyOutDir: true,
    rollupOptions: { input: "landing.html" },
  },
});
