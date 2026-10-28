export type Signal = "bm25" | "semantic" | "legal" | "citation" | "outcome";

export const SIGNALS: Signal[] = ["bm25", "semantic", "legal", "citation", "outcome"];

export const SIGNAL_LABEL: Record<Signal, string> = {
  bm25: "Lexical (BM25)",
  semantic: "Semantic (Doc2Vec)",
  legal: "Legal / statutory",
  citation: "Citation",
  outcome: "Outcome",
};

export const INTENT_LABEL: Record<string, string> = {
  FACTUAL_SIMILARITY: "Factual similarity",
  LEGAL_ISSUE: "Legal issue",
  STATUTE_BASED: "Statute-based",
  SUPPORTING_PRECEDENT: "Supporting precedent",
  CONTRASTING_PRECEDENT: "Contrasting precedent",
};

export interface QueryAnalysis {
  text: string;
  intent: string;
  intent_scores: Record<string, number>;
  intent_source: string;
  acts: Record<string, string>;
  provisions: string[];
  provision_labels: Record<string, string>;
  implied_provisions: string[];
  concepts: string[];
  precedents: string[];
  precedent_names: string[];
  outcome_requirements: string[];
}

export interface Explanation {
  matched_acts: { act: string; where: string }[];
  matched_provisions: { provision: string; where: string; implied: boolean }[];
  matched_concepts: string[];
  citation: { mode?: string; authority?: string; treatment?: string; authorities?: { name: string; treatment: string }[] };
  outcome: { procedural: string | null; result_text: string; substantive: Record<string, number> };
  contributions: Partial<Record<Signal, number>>;
  summary: string;
}

export interface SearchResult {
  rank: number;
  case_id: string;
  case_name: string;
  neutral_citation: string;
  scr_citation: string | null;
  decision_date: string | null;
  bench: string[];
  result: string | null;
  score: number;
  signals: Record<Signal, number>;
  normalized: Record<Signal, number> | null;
  issue: string;
  passage: string;
  explanation: Explanation;
}

export interface SearchResponse {
  query: QueryAnalysis;
  system: string;
  weights_name: string | null;
  semantic_variant: string;
  effective_weights: Partial<Record<Signal, number>> | null;
  applicable: Record<Signal, boolean>;
  timings_ms: Record<string, number>;
  results: SearchResult[];
}

export interface Meta {
  n_judgments: number;
  intents: string[];
  systems: Record<string, string>;
  defaults: Record<string, string>;
  selected_config: Record<string, unknown>;
  adaptive_weights: Record<string, Record<Signal, number>>;
  fixed_weights: Record<Signal, number>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep status text */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  search: (body: { query: string; system: string; top_k?: number; intent_override?: string | null }) =>
    request<SearchResponse>("/api/search", { method: "POST", body: JSON.stringify(body) }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  case: (id: string) => request<any>(`/api/cases/${encodeURIComponent(id)}`),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cases: (q: string, offset = 0) => request<any>(`/api/cases?q=${encodeURIComponent(q)}&offset=${offset}&limit=50`),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  evaluation: () => request<any>("/api/evaluation"),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  recompute: () => request<any>("/api/evaluation/recompute", { method: "POST" }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  recomputeStatus: () => request<any>("/api/evaluation/recompute"),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  labels: () => request<any>("/api/labels"),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  labelQuery: (qid: string) => request<any>(`/api/labels/${qid}`),
  setLabel: (qid: string, caseId: string, grade: number, note = "") =>
    request<{ ok: boolean }>(`/api/labels/${qid}/${encodeURIComponent(caseId)}`, {
      method: "PUT",
      body: JSON.stringify({ grade, note }),
    }),
  confirmAll: (qid: string) => request<{ ok: boolean; reviewed: number }>(`/api/labels/${qid}/confirm-all`, { method: "POST" }),
};
