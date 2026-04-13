export type ProgressSummary = {
  gather_round: number;
  plan_steps: number;
  evidence_items: number;
  has_report: boolean;
  need_more: boolean | undefined;
  verification_passed?: boolean | undefined;
  revision_count?: number | undefined;
};

export type ResearchResult = {
  thread_id: string;
  user_query: string | undefined;
  report: string | undefined;
  errors: string[];
  plan: unknown[];
  evidence: unknown[];
  reflection_rationale: string | undefined;
  gather_count: number | undefined;
  max_iterations: number | undefined;
  verification_passed?: boolean | undefined;
  verification_notes?: string | undefined;
  revision_count?: number | undefined;
};

export type StreamEvent =
  | { type: "started"; thread_id: string }
  | { type: "progress"; thread_id: string; summary: ProgressSummary }
  | { type: "complete"; thread_id: string; result: ResearchResult }
  | { type: "error"; thread_id: string; message: string };

export type ChatTurn = {
  id: string;
  query: string;
  status: "streaming" | "done" | "error";
  progress: ProgressSummary[];
  result?: ResearchResult;
  errorMessage?: string;
};
