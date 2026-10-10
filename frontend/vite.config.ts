import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  // Served locally; the guidebook's lessons and diagrams make the main bundle ~650 kB.
  build: { chunkSizeWarningLimit: 900 },
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
