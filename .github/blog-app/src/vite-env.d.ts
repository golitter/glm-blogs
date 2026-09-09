/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** China-time build stamp injected by the deploy workflow. */
  readonly VITE_BUILD_TIME?: string;
}
