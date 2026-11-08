import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages serves the site under /<repo>/
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
