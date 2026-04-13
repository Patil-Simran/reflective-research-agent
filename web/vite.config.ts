import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

const webRoot = path.dirname(fileURLToPath(import.meta.url));
const mermaidEntry = path.resolve(webRoot, "node_modules/mermaid/dist/mermaid.core.mjs");

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, webRoot, "");
  const proxyTarget =
    env.VITE_DEV_PROXY_TARGET?.trim() || "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    resolve: {
      // Explicit entry avoids occasional Vite import-analysis failures on package "exports".
      alias: [{ find: "mermaid", replacement: mermaidEntry }],
    },
    optimizeDeps: {
      include: ["mermaid"],
    },
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
