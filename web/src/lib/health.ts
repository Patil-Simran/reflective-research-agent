import { apiUrl } from "./apiBase";

export type HealthReadyJson = {
  ready?: boolean;
  components?: { ollama?: { ok?: boolean; detail?: string | null } };
};

function parseHealthReadyJson(raw: string): { json: HealthReadyJson | null; parseError: string } {
  const trimmed = raw.trim();
  if (!trimmed) {
    return { json: null, parseError: "empty response body" };
  }
  try {
    return { json: JSON.parse(trimmed) as HealthReadyJson, parseError: "" };
  } catch {
    return { json: null, parseError: "not valid JSON (proxy or HTML error page?)" };
  }
}

/**
 * Poll readiness (API cold-start + Ollama can take 1–2 minutes after `research serve`).
 */
export async function fetchBackendReady(maxAttempts = 12): Promise<{
  ok: boolean;
  detail: string;
}> {
  let detail = "";
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const r = await fetch(apiUrl("/api/health/ready"));
      const text = await r.text();
      const { json, parseError } = parseHealthReadyJson(text);

      if (!json) {
        detail = [
          `HTTP ${r.status} — ${parseError}.`,
          "Start the API from the project root: research serve (port 8000).",
          "If the UI is on another machine, set VITE_API_BASE_URL or VITE_DEV_PROXY_TARGET.",
        ].join(" ");
      } else if (json.ready === true) {
        return { ok: true, detail: "" };
      } else {
        detail =
          json.components?.ollama?.detail ??
          "Backend responded but is not ready (e.g. Ollama unreachable). Open GET /api/health/ready for details.";
      }
    } catch (e) {
      detail =
        e instanceof Error
          ? e.message
          : "Cannot reach the API (network error — is research serve running?)";
    }
    if (attempt < maxAttempts - 1) {
      await new Promise((resolve) => setTimeout(resolve, 800 * (attempt + 1)));
    }
  }
  return { ok: false, detail };
}
