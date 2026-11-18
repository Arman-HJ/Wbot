import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  // Avoid SPA HTML fallback for missing /packs/*.json (would break fetch().json()).
  appType: "mpa",
  plugins: [
    viteStaticCopy({
      targets: [
        {
          // mujoco-js embeds wasm in JS; ORT needs wasm + companion .mjs loaders.
          src: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*",
          dest: "ort",
        },
      ],
    }),
  ],
  server: {
    port: 5173,
    host: true,
    headers: {
      // Needed for onnxruntime-web threaded wasm (SharedArrayBuffer).
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    exclude: ["mujoco-js"],
  },
  assetsInclude: ["**/*.wasm", "**/*.onnx"],
});
