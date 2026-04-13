/** Curated starters aligned with plan → gather → reflect → brief → verify flow. */

export type SuggestionTone = "violet" | "cyan" | "amber" | "emerald" | "fuchsia";

export type Suggestion = {
  label: string;
  query: string;
  tone: SuggestionTone;
};

export type SuggestionGroup = {
  title: string;
  subtitle?: string;
  items: Suggestion[];
};

const toneChip: Record<SuggestionTone, string> = {
  violet:
    "border-violet-500/35 bg-violet-500/[0.08] text-violet-100 hover:border-violet-400/55 hover:bg-violet-500/15",
  cyan: "border-cyan-500/35 bg-cyan-500/[0.08] text-cyan-100 hover:border-cyan-400/55 hover:bg-cyan-500/15",
  amber:
    "border-amber-500/35 bg-amber-500/[0.08] text-amber-100 hover:border-amber-400/55 hover:bg-amber-500/15",
  emerald:
    "border-emerald-500/35 bg-emerald-500/[0.08] text-emerald-100 hover:border-emerald-400/55 hover:bg-emerald-500/15",
  fuchsia:
    "border-fuchsia-500/35 bg-fuchsia-500/[0.08] text-fuchsia-100 hover:border-fuchsia-400/55 hover:bg-fuchsia-500/15",
};

export function suggestionChipClass(tone: SuggestionTone): string {
  return `rounded-xl border px-3 py-2 text-left text-xs font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-500/45 ${toneChip[tone]}`;
}

export const SUGGESTION_GROUPS: SuggestionGroup[] = [
  {
    title: "Compare & decide",
    subtitle: "Surfaces trade-offs, tables, and citations",
    items: [
      {
        label: "RAG vs fine-tuning",
        query:
          "Compare retrieval-augmented generation versus supervised fine-tuning for domain-specific Q&A: when to use each, cost, latency, and failure modes.",
        tone: "violet",
      },
      {
        label: "Sync vs async APIs",
        query:
          "Compare synchronous request–response APIs with asynchronous messaging for distributed systems: trade-offs, consistency, and operational complexity.",
        tone: "cyan",
      },
      {
        label: "SQL vs NoSQL",
        query:
          "When should teams choose relational SQL versus document or wide-column stores? Compare consistency models, scaling patterns, and migration risk.",
        tone: "amber",
      },
    ],
  },
  {
    title: "ML & inference",
    subtitle: "Web + arXiv-style evidence",
    items: [
      {
        label: "Quantized LLM inference",
        query:
          "Summarize recent approaches to quantized large language model inference: PTQ vs QAT, KV cache effects, and accuracy–speed trade-offs.",
        tone: "fuchsia",
      },
      {
        label: "Speculative decoding",
        query:
          "Explain speculative decoding for transformer inference: draft models, acceptance criteria, and when it helps throughput.",
        tone: "violet",
      },
      {
        label: "Embedding models",
        query:
          "How do text embedding models work for RAG? Contrast sentence-transformers style bi-encoders with late interaction and scaling to long documents.",
        tone: "emerald",
      },
    ],
  },
  {
    title: "Architecture & diagrams",
    subtitle: "Encourages Mermaid / figures in the report",
    items: [
      {
        label: "Event-driven flow",
        query:
          "Describe an event-driven microservices architecture with a diagram-friendly breakdown: brokers, sagas, idempotency, and failure handling.",
        tone: "cyan",
      },
      {
        label: "Load balancer path",
        query:
          "Walk through the request path from client through DNS, load balancer, and app tier; include sequence-style explanation suitable for a diagram.",
        tone: "amber",
      },
    ],
  },
  {
    title: "RAG & production",
    subtitle: "Chunking, eval, and ops",
    items: [
      {
        label: "RAG failure modes",
        query:
          "What breaks RAG pipelines in production? Cover chunking, embedding drift, stale corpora, hallucinated citations, and mitigations.",
        tone: "emerald",
      },
      {
        label: "Chunking strategies",
        query:
          "Compare fixed-size, sentence-aware, and hierarchical chunking for vector retrieval; include practical defaults and evaluation ideas.",
        tone: "violet",
      },
      {
        label: "Hybrid search",
        query:
          "Explain hybrid sparse–dense retrieval for RAG: BM25 plus embeddings, fusion methods, and when hybrid beats dense-only.",
        tone: "fuchsia",
      },
    ],
  },
];

/** Compact row after first turn — fast follow-ups */
export const FOLLOW_UP_CHIPS: Suggestion[] = [
  {
    label: "Deeper dive",
    query: "Go deeper on the main mechanism above: edge cases, limits, and one concrete example.",
    tone: "violet",
  },
  {
    label: "Add comparison table",
    query: "Revisit the last topic and add a Markdown comparison table of options with pros, cons, and citations.",
    tone: "cyan",
  },
  {
    label: "Security angle",
    query: "From a security and abuse perspective, what are the main risks and mitigations for this topic?",
    tone: "amber",
  },
  {
    label: "Smaller scope",
    query: "Narrow to a minimal MVP or smallest useful version of this idea with clear steps.",
    tone: "emerald",
  },
];

/** Diagram / table / gaps — same handlers as FOLLOW_UP_CHIPS (full chip hit target). */
export const FOLLOW_UP_TOOL_CHIPS: Suggestion[] = [
  {
    label: "+ Diagram prompt",
    query:
      "Add a Mermaid flowchart or sequence diagram for the main process described in your last answer in this session. Base it on that report and its citations.",
    tone: "cyan",
  },
  {
    label: "+ Table prompt",
    query:
      "Add a Markdown table comparing the main options from your last answer in this session with pros, cons, and citations.",
    tone: "amber",
  },
  {
    label: "+ Gaps prompt",
    query:
      "List limitations of the evidence in your last answer in this session and what you would search next to strengthen it.",
    tone: "emerald",
  },
];
