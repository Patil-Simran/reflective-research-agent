import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatTurn, ProgressSummary, StreamEvent } from "./types";
import { buildContextualResearchQuery } from "./lib/contextualQuery";
import { fetchBackendReady } from "./lib/health";
import {
  FOLLOW_UP_CHIPS,
  FOLLOW_UP_TOOL_CHIPS,
  SUGGESTION_GROUPS,
  suggestionChipClass,
} from "./lib/suggestions";
import { streamResearch } from "./lib/stream";
import { MarkdownBody } from "./components/MarkdownBody";
import { ProgressStrip } from "./components/ProgressStrip";
import { normalizeReferenceUrl, referencesFromEvidence } from "./lib/evidence";

function newId() {
  return crypto.randomUUID();
}

type BackendGate = "loading" | "ok" | "down";

export default function App() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [backendGate, setBackendGate] = useState<BackendGate>("loading");
  const [backendDetail, setBackendDetail] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const busyRef = useRef(false);
  /** Avoid stale `busy` in useCallback closures — chips must not no-op after a run finishes. */
  const researchInFlightRef = useRef(false);
  const turnsRef = useRef<ChatTurn[]>([]);
  const threadIdRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  /** Bumped on New session so in-flight SSE handlers cannot write after reset. */
  const sessionGenerationRef = useRef(0);

  const resetSession = useCallback(() => {
    sessionGenerationRef.current += 1;
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    threadIdRef.current = null;
    researchInFlightRef.current = false;
    turnsRef.current = [];
    setBusy(false);
    setTurns([]);
    setInput("");
  }, []);

  useEffect(() => {
    turnsRef.current = turns;
  }, [turns]);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const res = await fetchBackendReady();
      if (cancelled) return;
      setBackendGate(res.ok ? "ok" : "down");
      setBackendDetail(res.detail);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      sessionGenerationRef.current += 1;
      streamAbortRef.current?.abort();
    };
  }, []);

  const recheckBackend = useCallback(() => {
    setBackendGate("loading");
    setBackendDetail("");
    void (async () => {
      const res = await fetchBackendReady();
      setBackendGate(res.ok ? "ok" : "down");
      setBackendDetail(res.detail);
    })();
  }, []);

  const scrollToBottom = () => {
    requestAnimationFrame(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }));
  };

  const appendProgress = useCallback((turnId: string, summary: ProgressSummary) => {
    setTurns((prev) =>
      prev.map((t) =>
        t.id === turnId ? { ...t, progress: [...t.progress, summary] } : t,
      ),
    );
    scrollToBottom();
  }, []);

  const runQuery = useCallback(
    async (q: string) => {
      const trimmed = q.trim();
      if (!trimmed || backendGate !== "ok") return;
      // Use busyRef so we never read a stale `busy` from a closure (would skip heal and block forever).
      if (!busyRef.current && researchInFlightRef.current) {
        researchInFlightRef.current = false;
      }
      if (researchInFlightRef.current) return;
      researchInFlightRef.current = true;

      const genAtStart = sessionGenerationRef.current;

      if (threadIdRef.current === null) {
        threadIdRef.current = newId();
      }
      const sessionThreadId = threadIdRef.current;

      const priorReport = [...turnsRef.current]
        .reverse()
        .find((t) => t.status === "done" && t.result?.report)?.result?.report;
      const payload = buildContextualResearchQuery(trimmed, priorReport);

      const turnId = newId();
      setTurns((prev) => [...prev, { id: turnId, query: trimmed, status: "streaming", progress: [] }]);
      setInput("");
      setBusy(true);
      scrollToBottom();

      streamAbortRef.current?.abort();
      const ac = new AbortController();
      streamAbortRef.current = ac;

      const handle = (ev: StreamEvent) => {
        if (genAtStart !== sessionGenerationRef.current) {
          return;
        }
        if (ev.type === "started" && ev.thread_id) {
          threadIdRef.current = ev.thread_id;
        }
        if (ev.type === "progress") {
          appendProgress(turnId, ev.summary);
        }
        if (ev.type === "complete") {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === turnId
                ? { ...t, status: "done", result: ev.result, progress: t.progress }
                : t,
            ),
          );
        }
        if (ev.type === "error") {
          setTurns((prev) =>
            prev.map((t) =>
              t.id === turnId ? { ...t, status: "error", errorMessage: ev.message } : t,
            ),
          );
        }
      };

      try {
        await streamResearch(payload, sessionThreadId, handle, { signal: ac.signal });
      } catch (e) {
        const aborted =
          (typeof DOMException !== "undefined" &&
            e instanceof DOMException &&
            e.name === "AbortError") ||
          (e instanceof Error && e.name === "AbortError");
        if (aborted) {
          return;
        }
        if (genAtStart !== sessionGenerationRef.current) {
          return;
        }
        setTurns((prev) =>
          prev.map((t) =>
            t.id === turnId
              ? { ...t, status: "error", errorMessage: e instanceof Error ? e.message : String(e) }
              : t,
          ),
        );
      } finally {
        if (streamAbortRef.current === ac) {
          streamAbortRef.current = null;
        }
        if (genAtStart === sessionGenerationRef.current) {
          researchInFlightRef.current = false;
          setBusy(false);
        }
        scrollToBottom();
      }
    },
    [appendProgress, backendGate],
  );

  const fillPromptAndFocus = useCallback((text: string) => {
    setInput(text);
    requestAnimationFrame(() => {
      const el = promptRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(text.length, text.length);
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }, []);

  const submit = () => {
    void runQuery(input);
  };

  return (
    <div className="app-bg flex min-h-screen flex-col text-zinc-100">
      <header className="glass sticky top-0 z-50 border-b border-zinc-800/80">
        <div className="relative z-50 mx-auto flex max-w-4xl items-center justify-between gap-3 px-4 py-3">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 shadow-lg shadow-violet-500/20">
              <svg className="h-5 w-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 19V5l8 7 8-7v14"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-lg font-semibold tracking-tight text-white">
                Reflective Research
              </h1>
              {backendGate === "loading" && (
                <p className="text-[11px] text-zinc-500">Checking backend…</p>
              )}
            </div>
          </div>
          <div className="pointer-events-auto relative z-[60] flex shrink-0 items-center gap-2">
            <button
              type="button"
              title={
                turns.length === 0 && !busy
                  ? "Reset session (nothing to clear)"
                  : "Stop stream if running, clear chat, reset thread"
              }
              aria-label="New session: clear chat and reset research thread"
              onClick={() => resetSession()}
              className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-zinc-500 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/50"
            >
              New session
            </button>
          </div>
        </div>
      </header>

      {backendGate === "down" && (
        <div className="border-b border-red-900/60 bg-red-950/90 px-4 py-2 text-center text-sm text-red-100">
          <p className="font-medium">Backend not ready — check API and Ollama, then retry or refresh.</p>
          {backendDetail && (
            <pre className="mx-auto mt-2 max-h-28 max-w-3xl overflow-auto whitespace-pre-wrap text-left text-[11px] text-red-300/90">
              {backendDetail}
            </pre>
          )}
          <button
            type="button"
            onClick={() => recheckBackend()}
            className="mt-3 rounded-lg border border-red-400/40 bg-red-900/40 px-4 py-1.5 text-xs font-medium text-red-50 transition hover:bg-red-900/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-300/60"
          >
            Retry connection
          </button>
        </div>
      )}

      <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 px-4 py-8 pb-40">
        {turns.length === 0 && (
          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6 sm:p-8">
            <div className="text-center">
              <p className="text-lg font-medium text-zinc-200">What should we research?</p>
              <p className="mx-auto mt-2 max-w-lg text-sm text-zinc-500">
                Pick a topic to run the full pipeline — plan, gather (web + optional RAG), brief,
                cited report, and verification.
              </p>
            </div>
            <div className="mx-auto mt-8 max-w-2xl space-y-8">
              {SUGGESTION_GROUPS.map((group) => (
                <section key={group.title}>
                  <div className="flex flex-wrap items-end justify-between gap-2 border-b border-zinc-800/80 pb-2">
                    <h2 className="text-sm font-semibold tracking-tight text-zinc-200">{group.title}</h2>
                    {group.subtitle && (
                      <span className="text-[11px] text-zinc-500">{group.subtitle}</span>
                    )}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {group.items.map((s) => (
                      <button
                        key={s.label}
                        type="button"
                        disabled={busy || backendGate !== "ok"}
                        title={s.query}
                        aria-label={`Run research: ${s.query}`}
                        onClick={() => void runQuery(s.query)}
                        className={suggestionChipClass(s.tone)}
                      >
                        {s.label}
                      </button>
                    ))}
                  </div>
                </section>
              ))}
            </div>
            <p className="mt-6 text-center text-[11px] text-zinc-600">
              Starters run immediately; use the box below for your own question.
            </p>
          </div>
        )}

        {turns.map((t) => (
          <article key={t.id} className="space-y-3">
            <div className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-zinc-800 px-4 py-3 text-sm leading-relaxed text-zinc-100 ring-1 ring-zinc-700">
                {t.query}
              </div>
            </div>

            <div className="flex justify-start">
              <div className="max-w-[min(100%,42rem)] w-full rounded-2xl rounded-bl-md border border-zinc-800 bg-zinc-900/60 px-5 py-4 shadow-xl shadow-black/20 ring-1 ring-zinc-800/80">
                {t.status === "streaming" && (
                  <>
                    <ProgressStrip items={t.progress} active />
                  </>
                )}

                {t.status === "error" && (
                  <div className="rounded-lg border border-red-900/50 bg-red-950/40 px-3 py-2 text-sm text-red-200">
                    {t.errorMessage ?? "Unknown error"}
                  </div>
                )}

                {t.status === "done" && t.result && (
                  <>
                    <ProgressStrip
                      items={t.progress}
                      active={false}
                    />
                    {t.result.errors.length > 0 && (
                      <ul className="mb-4 list-inside list-disc text-xs text-amber-200/90">
                        {t.result.errors.map((err) => (
                          <li key={err}>{err}</li>
                        ))}
                      </ul>
                    )}
                    {t.result.reflection_rationale && (
                      <details className="mb-4 rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-xs text-zinc-400">
                        <summary className="cursor-pointer font-medium text-zinc-300">
                          Reflection note
                        </summary>
                        <p className="mt-2 whitespace-pre-wrap">{t.result.reflection_rationale}</p>
                      </details>
                    )}
                    {t.result.verification_notes && (
                      <details className="mb-4 rounded-lg border border-zinc-800 bg-zinc-950/50 p-3 text-xs text-zinc-400">
                        <summary className="cursor-pointer font-medium text-zinc-300">
                          Verification
                          {t.result.verification_passed === true
                            ? " (passed)"
                            : t.result.verification_passed === false
                              ? " (issues)"
                              : ""}
                        </summary>
                        <p className="mt-2 whitespace-pre-wrap">{t.result.verification_notes}</p>
                        {typeof t.result.revision_count === "number" && t.result.revision_count > 0 && (
                          <p className="mt-2 text-zinc-500">
                            Revision rounds applied: {t.result.revision_count}
                          </p>
                        )}
                      </details>
                    )}
                    <MarkdownBody content={t.result.report ?? "_Empty report._"} />
                    {(() => {
                      const refs = referencesFromEvidence(t.result.evidence);
                      if (refs.length === 0) return null;
                      return (
                        <details className="mt-6 border-t border-zinc-800 pt-4" open>
                          <summary className="cursor-pointer text-xs font-medium text-zinc-400">
                            Sources cited in report ({refs.length})
                          </summary>
                          <ul className="mt-3 space-y-3">
                            {refs.map((r) => (
                              <li
                                key={`${r.citeNum}-${r.ref.slice(0, 48)}`}
                                className="rounded-lg border border-zinc-800/80 bg-zinc-950/40 p-3 text-xs"
                              >
                                <div className="flex flex-wrap items-baseline gap-2 text-zinc-300">
                                  <span className="font-mono text-[10px] text-violet-400/90">
                                    [{r.citeNum}]
                                  </span>
                                  <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                                    {r.sourceType}
                                  </span>
                                </div>
                                <p className="mt-1 font-medium text-zinc-200">{r.title}</p>
                                {r.ref !== "—" && (() => {
                                  const href = normalizeReferenceUrl(r.ref);
                                  const openable = /^https?:\/\//i.test(href);
                                  return (
                                    <p className="mt-0.5 break-all text-[11px] text-cyan-500/90">
                                      {openable ? (
                                        <a
                                          href={href}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="underline decoration-cyan-500/40 hover:decoration-cyan-400"
                                        >
                                          {href}
                                        </a>
                                      ) : (
                                        r.ref
                                      )}
                                    </p>
                                  );
                                })()}
                                <p className="mt-2 whitespace-pre-wrap text-[11px] leading-relaxed text-zinc-500">
                                  {r.preview}
                                </p>
                              </li>
                            ))}
                          </ul>
                        </details>
                      );
                    })()}
                  </>
                )}
              </div>
            </div>
          </article>
        ))}
        <div ref={bottomRef} />
      </main>

      <div className="glass fixed bottom-0 left-0 right-0 isolate z-[60] border-t border-zinc-800/80 px-4 pb-4 pt-3">
        {turns.length > 0 && backendGate === "ok" && (
          <div className="relative z-10 mx-auto mb-3 max-w-4xl pointer-events-auto">
            <p className="mb-2 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-zinc-500">
              Quick follow-up
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {FOLLOW_UP_CHIPS.map((s) => (
                <button
                  key={s.label}
                  type="button"
                  disabled={busy || backendGate !== "ok"}
                  title={s.query}
                  aria-label={`Follow-up: ${s.query}`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    void runQuery(s.query);
                  }}
                  className={`${suggestionChipClass(s.tone)} max-w-[11rem] touch-manipulation sm:max-w-none`}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <div className="mt-2 border-t border-zinc-800/60 pt-2">
              <p className="mb-2 text-center text-[10px] text-zinc-500">
                + row: insert prompt in the box — then press <span className="text-zinc-400">Run</span>
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {FOLLOW_UP_TOOL_CHIPS.map((s) => (
                  <button
                    key={s.label}
                    type="button"
                    disabled={busy || backendGate !== "ok"}
                    title={`${s.query}\n\n(Inserts into the text box; press Run.)`}
                    aria-label={`Insert prompt: ${s.query}`}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      fillPromptAndFocus(s.query);
                    }}
                    className={`${suggestionChipClass(s.tone)} max-w-[11rem] touch-manipulation sm:max-w-none`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        <form
          className="mx-auto flex max-w-4xl gap-3"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            ref={promptRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void submit();
              }
            }}
            rows={2}
            placeholder="Ask a research question… (Shift+Enter for newline)"
            disabled={busy || backendGate !== "ok"}
            className="min-h-[52px] flex-1 resize-y rounded-xl border border-zinc-700 bg-zinc-950/80 px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/20 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={busy || !input.trim() || backendGate !== "ok"}
            aria-busy={busy}
            className="self-end rounded-xl bg-gradient-to-r from-violet-600 to-cyan-500 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-600/25 transition hover:brightness-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? "…" : "Run"}
          </button>
        </form>
      </div>
    </div>
  );
}
