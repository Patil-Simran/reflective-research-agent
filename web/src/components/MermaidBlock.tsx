import { useEffect, useId, useRef, useState } from "react";
import mermaid from "mermaid";

let configured = false;

function ensureMermaid() {
  if (configured) return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    securityLevel: "strict",
    fontFamily: "ui-sans-serif, system-ui, sans-serif",
  });
  configured = true;
}

/**
 * Common LLM mistakes in flowchart edge labels, e.g. `-->|Request|> B` instead of `-->|Request| B`.
 */
function sanitizeMermaidSource(raw: string): string {
  let s = raw.trim();
  // `|label|>` before next token → `|label|`
  s = s.replace(/\|([^|\n]+)\|>(?=\s)/g, "|$1|");
  return s;
}

type Props = { chart: string };

export function MermaidBlock({ chart }: Props) {
  const reactId = useId().replace(/[^a-zA-Z0-9]/g, "");
  const host = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    ensureMermaid();
    const el = host.current;
    if (!el) return;
    let cancelled = false;
    const baseId = `g-${reactId}-${Math.random().toString(36).slice(2, 10)}`;

    const tryRender = async (source: string, suffix: string) => {
      const { svg } = await mermaid.render(`${baseId}${suffix}`, source);
      if (!cancelled && el) {
        el.innerHTML = svg;
        setErr(null);
      }
    };

    void (async () => {
      const original = chart.trim();
      const sanitized = sanitizeMermaidSource(chart);

      try {
        await tryRender(original, "");
        return;
      } catch (first) {
        if (cancelled) return;
        if (sanitized !== original) {
          try {
            await tryRender(sanitized, "-fix");
            return;
          } catch {
            if (!cancelled) {
              setErr(first instanceof Error ? first.message : "parse error");
              el.innerHTML = "";
            }
            return;
          }
        }
        if (!cancelled) {
          setErr(first instanceof Error ? first.message : "parse error");
          el.innerHTML = "";
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [chart, reactId]);

  if (err) {
    return (
      <div className="my-4 rounded-lg border border-amber-900/40 bg-amber-950/20 p-3">
        <p className="mb-2 text-[11px] font-medium text-amber-200/95">
          Mermaid could not be rendered (invalid or unsupported syntax).
        </p>
        <p className="mb-2 text-[10px] text-zinc-500">
          No external images are loaded for diagrams. Fix the source below or paste it into a Mermaid
          editor to debug.
        </p>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border border-zinc-800 bg-zinc-950 p-3 font-mono text-[11px] text-zinc-300">
          {chart}
        </pre>
      </div>
    );
  }

  return (
    <div
      ref={host}
      className="my-4 flex justify-center overflow-x-auto rounded-lg border border-zinc-800 bg-zinc-950/80 p-4 [&_svg]:max-h-[480px] [&_svg]:max-w-full"
    />
  );
}
