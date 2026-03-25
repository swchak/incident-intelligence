import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.js",
    globals: true
  },
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});
