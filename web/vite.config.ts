import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset paths so the build can be served from any sub-path (e.g. GitHub Pages).
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 1024 },
});
