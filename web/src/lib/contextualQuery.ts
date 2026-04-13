const MAX_PRIOR_CHARS = 3600;

/**
 * Prefix the latest completed report so follow-ups (same session) carry explicit context
 * for the model, without changing what the user sees in the chat bubble.
 */
export function buildContextualResearchQuery(
  userRequest: string,
  priorReportMarkdown: string | undefined,
): string {
  const t = userRequest.trim();
  const prior = priorReportMarkdown?.trim();
  if (!prior) return t;
  const excerpt = prior.slice(0, MAX_PRIOR_CHARS);
  return (
    "[Prior completed report in this session — use as grounding, then answer the follow-up.]\n\n" +
    `${excerpt}\n\n` +
    `[Follow-up request]\n${t}`
  );
}
