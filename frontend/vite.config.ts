import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // strictPort matters: the FastAPI CORS allowlist only permits 5173. Without
  // it Vite silently moves to 5174 when 5173 is busy and every API call is
  // then blocked by the browser. Better to fail loudly and free the port.
  server: { port: 5173, strictPort: true },
});
