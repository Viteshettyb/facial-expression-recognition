/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the FastAPI analysis service, e.g. http://127.0.0.1:8000 */
  readonly VITE_API_URL?: string;
  /** Legacy alias for VITE_API_URL. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
