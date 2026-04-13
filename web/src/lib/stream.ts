import type { StreamEvent } from "../types";
import { apiUrl } from "./apiBase";

export type StreamResearchOptions = {
  signal?: AbortSignal;
};

/**
 * POST SSE: read `data: {...}\n\n` frames from a fetch body stream.
 * Pass `signal` to cancel (e.g. New session); surfaces `AbortError` to the caller.
 */
export async function streamResearch(
  query: string,
  threadId: string | undefined,
  onEvent: (ev: StreamEvent) => void,
  options?: StreamResearchOptions,
): Promise<void> {
  const res = await fetch(apiUrl("/api/research/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      thread_id: threadId ?? null,
    }),
    signal: options?.signal,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  const flushBlock = (block: string) => {
    const lines = block.split("\n");
    for (const line of lines) {
      if (line.startsWith("data:")) {
        const raw = line.slice(5).trim();
        if (!raw) continue;
        try {
          onEvent(JSON.parse(raw) as StreamEvent);
        } catch {
          console.warn("Bad SSE JSON", raw);
        }
      }
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const p of parts) {
        if (p.trim()) flushBlock(p);
      }
    }
    if (buffer.trim()) flushBlock(buffer);
  } finally {
    reader.releaseLock();
  }
}
