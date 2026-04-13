/**
 * API origin for fetch(). Empty string = relative URLs (Vite dev proxy or same-origin deploy).
 */
const raw = import.meta.env.VITE_API_BASE_URL as string | undefined;
export const API_BASE = raw?.replace(/\/$/, "").trim() ?? "";

export function apiUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  return API_BASE ? `${API_BASE}${p}` : p;
}
