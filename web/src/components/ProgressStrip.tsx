import type { ProgressSummary } from "../types";

type Props = { items: ProgressSummary[]; active: boolean };

export function ProgressStrip({ items, active }: Props) {
  const last = items.length ? items[items.length - 1] : null;
  if (!last && !active) return null;

  return (
    <div className="mt-3 space-y-2">
      {active && (
        <div className="h-1 rounded-full overflow-hidden bg-zinc-800">
          <div className="h-full w-1/3 rounded-full shimmer-bar" />
        </div>
      )}
      {last && (
        <div className="flex flex-wrap gap-2 text-xs font-medium">
          <span className="rounded-full bg-violet-500/15 px-2.5 py-1 text-violet-200 ring-1 ring-violet-500/30">
            Gather {last.gather_round}
          </span>
          <span className="rounded-full bg-zinc-800 px-2.5 py-1 text-zinc-300 ring-1 ring-zinc-700">
            Plan {last.plan_steps} steps
          </span>
          <span className="rounded-full bg-cyan-500/10 px-2.5 py-1 text-cyan-200 ring-1 ring-cyan-500/25">
            Evidence {last.evidence_items}
          </span>
          {last.has_report && !active && (
            <span className="rounded-full bg-emerald-500/15 px-2.5 py-1 text-emerald-200 ring-1 ring-emerald-500/30">
              Report in state
            </span>
          )}
          {last.need_more === true && (
            <span className="rounded-full bg-amber-500/15 px-2.5 py-1 text-amber-200 ring-1 ring-amber-500/30">
              Reflecting…
            </span>
          )}
          {typeof last.revision_count === "number" && last.revision_count > 0 && (
            <span className="rounded-full bg-fuchsia-500/15 px-2.5 py-1 text-fuchsia-200 ring-1 ring-fuchsia-500/30">
              Revise {last.revision_count}
            </span>
          )}
          {last.verification_passed === true && !active && (
            <span className="rounded-full bg-emerald-500/15 px-2.5 py-1 text-emerald-200 ring-1 ring-emerald-500/30">
              Verified
            </span>
          )}
          {last.verification_passed === false && !active && (
            <span className="rounded-full bg-orange-500/15 px-2.5 py-1 text-orange-200 ring-1 ring-orange-500/30">
              Verify issues
            </span>
          )}
        </div>
      )}
    </div>
  );
}
