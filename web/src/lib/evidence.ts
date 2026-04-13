/** Evidence rows worth showing as references (exclude empty-RAG / failed-search placeholders). */

export type ReferenceRow = {
  /** 1-based index matching in-text [1], [2], … */
  citeNum: number;
  id: string;
  sourceType: string;
  ref: string;
  title: string;
  preview: string;
};

const EMPTY_MARKERS = [
  "no web results returned",
  "rag returned no chunks",
  "search failed:",
  "no documents loaded",
];

/** Model-invented legacy tags in report body (esp. fake "References" sections). */
const LEGACY_E_TAG_RE = /\[\s*E\s*:\s*[^\]]+\]/gi;

export function stripLegacyEvidenceTags(text: string): string {
  return text.replace(LEGACY_E_TAG_RE, "").replace(/[ \t]+\n/g, "\n");
}

function cleanDisplayText(s: string): string {
  return stripLegacyEvidenceTags(s).replace(/\s+/g, " ").trim();
}

function isPlaceholderContent(content: string): boolean {
  const low = content.toLowerCase();
  return EMPTY_MARKERS.some((m) => low.includes(m));
}

/** First clean https URL in ref (handles leading space, pasted prose, trailing punctuation). */
export function normalizeReferenceUrl(ref: string): string {
  const t = ref.trim();
  if (/^https?:\/\//i.test(t)) {
    const first = t.split(/\s/)[0] ?? t;
    return first.replace(/[),.;]+$/g, "");
  }
  const m = t.match(/https?:\/\/[^\s<>"'[\])]+/i);
  return m ? m[0].replace(/[),.;]+$/g, "") : t;
}

function writerCiteFromRow(e: Record<string, unknown>): number | null {
  const raw = e.writer_cite ?? e.cite;
  if (raw === undefined || raw === null) return null;
  const n = typeof raw === "number" ? raw : Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

export function referencesFromEvidence(evidence: unknown[]): ReferenceRow[] {
  const out: ReferenceRow[] = [];
  let seq = 1;
  for (const raw of evidence) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const st = e.source_type;
    if (st !== "search" && st !== "rag") continue;
    const content = String(e.content ?? "").trim();
    if (content.length < 50 || isPlaceholderContent(content)) continue;
    const meta =
      e.metadata && typeof e.metadata === "object"
        ? (e.metadata as Record<string, unknown>)
        : {};
    const titleRaw = String(meta.title ?? "").trim();
    const firstLine = content.split("\n")[0]?.trim() ?? "";
    const titleCombined =
      titleRaw || (firstLine.length > 0 ? firstLine.slice(0, 140) : "Source excerpt");
    const title = cleanDisplayText(titleCombined) || "Source excerpt";
    const refRaw = String(e.source_ref ?? "").trim() || "—";
    const ref = refRaw === "—" ? "—" : normalizeReferenceUrl(refRaw);
    const previewRaw = content.length > 360 ? `${content.slice(0, 357)}…` : content;
    const preview = stripLegacyEvidenceTags(previewRaw).trim();
    const wc = writerCiteFromRow(e);
    const citeNum = wc !== null ? wc : seq++;
    out.push({
      citeNum,
      id: String(e.id ?? ""),
      sourceType: String(st),
      ref,
      title,
      preview,
    });
  }
  out.sort((a, b) => a.citeNum - b.citeNum);
  return out;
}
