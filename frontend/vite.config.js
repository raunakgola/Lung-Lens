import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// API base URL comes from .env (VITE_API_URL). The proxy below is a fallback
// so same-origin fetches also work in dev if VITE_API_URL is unset.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/explain': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
