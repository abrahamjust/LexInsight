import { useEffect, useState } from "react";
import { api, INTENT_LABEL } from "../api";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const MAIN: [string, string][] = [
  ["B1_bm25", "B1 BM25"],
  ["B2_doc2vec_whole", "B2 Doc2Vec (whole)"],
  ["B3_doc2vec_chunked_max", "B3 Doc2Vec (chunked, max)"],
  ["B3b_doc2vec_chunked_mean3", "B3b Doc2Vec (chunked, mean-3)"],
  ["X_legal_only", "Legal scorer only"],
  ["B4_bm25_legal", "B4 BM25 + legal"],
  ["B5_bm25_semantic", "B5 BM25 + Doc2Vec"],
  ["B6_fixed_hybrid", "B6 Fixed-weight hybrid"],
  ["B7_adaptive_hybrid", "B7 Intent-adaptive hybrid"],
  ["FULL_oracle_intent", "B7 with gold intent"],
  ["FULL_rule_intent", "B7 with rule-based intent"],
];
const ABL: [string, string][] = [
  ["FULL_minus_bm25", "− BM25"], ["FULL_minus_semantic", "− semantic"], ["FULL_minus_legal", "− legal"],
  ["FULL_minus_citation", "− citation"], ["FULL_minus_outcome", "− outcome"], ["FULL_fixed_weights", "fixed weights"],
];
const METRICS = ["P@5", "R@5", "MRR@5", "nDCG@5", "nDCG@10", "judged@5"];
const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "—" : v.toFixed(d));

function MetricTable({ summary, rows, highlight }: { summary: Any; rows: [string, string][]; highlight?: string[] }) {
  const present = rows.filter(([k]) => summary[k]);
  const best: Record<string, number> = {};
  for (const m of METRICS) best[m] = Math.max(...present.slice(0, 9).map(([k]) => summary[k][m] ?? -1));
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>System</th>{METRICS.map((m) => <th key={m} className="num">{m}</th>)}</tr></thead>
        <tbody>
          {present.map(([k, label]) => (
            <tr key={k} className={highlight?.includes(k) ? "highlight" : ""}>
              <td>{label}</td>
              {METRICS.map((m) => (
                <td key={m} className={`num ${m !== "judged@5" && summary[k][m] === best[m] ? "best" : ""}`}>{fmt(summary[k][m])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function EvaluationPage() {
  const [data, setData] = useState<Any>(null);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<Any>(null);

  const load = () => api.evaluation().then(setData).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  async function recompute() {
    setJob(await api.recompute());
    const timer = setInterval(async () => {
      const s = await api.recomputeStatus();
      setJob(s);
      if (!s.running) { clearInterval(timer); load(); }
    }, 2000);
  }

  if (error) return <div className="banner bad">{error}</div>;
  if (!data) return <div className="row"><span className="spinner" /> Loading…</div>;
  const r = data.test_report;
  const ic = data.intent_classification;
  const ex = data.extraction_quality;

  return (
    <div>
      <div className="row">
        <h1 style={{ marginBottom: 0 }}>Evaluation</h1>
        <span className="spacer" />
        <button className="btn" onClick={recompute} disabled={job?.running}>
          {job?.running ? <><span className="spinner" /> Recomputing…</> : "Recompute metrics from current labels"}
        </button>
      </div>
      {job?.error && <div className="banner bad section">{job.error}</div>}

      {!r && <div className="banner section">No test report yet. Run the evaluation workflow (see README).</div>}
      {r && (
        <>
          <div className="grid-cols section">
            <div className="card stat"><b>{r.dataset.corpus.judgments_written}</b><span>judgments indexed</span></div>
            <div className="card stat"><b>{r.n_test_queries}</b><span>test queries ({r.n_test_queries_with_relevant} with a relevant judgment)</span></div>
            <div className="card stat"><b>{fmt(r.summary.B7_adaptive_hybrid?.["nDCG@5"])}</b><span>B7 nDCG@5 (vs B1 BM25 {fmt(r.summary.B1_bm25?.["nDCG@5"])})</span></div>
            <div className="card stat"><b>{r.efficiency.latency_ms_median.total_ms} ms</b><span>median end-to-end search</span></div>
          </div>

          <div className="card section">
            <h2>Retrieval quality on the test queries</h2>
            <p className="faint">Frozen configuration chosen on validation: semantic <code>{r.selected_configuration.semantic_variant}</code>,
              fixed <code>{r.selected_configuration.fixed}</code>, adaptive <code>{r.selected_configuration.adaptive}</code>. Bold = best among B1–B7.
              judged@5 is the share of top-5 results that have a label; unjudged results count as not relevant.</p>
            <MetricTable summary={r.summary} rows={MAIN} highlight={["B6_fixed_hybrid", "B7_adaptive_hybrid"]} />
          </div>

          <div className="card section">
            <h2>Paired comparisons</h2>
            <p className="faint">Difference in mean nDCG@5 over the same test queries, 95% paired-bootstrap CI and two-sided sign-flip randomisation p-value.</p>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Comparison</th><th className="num">Δ nDCG@5</th><th className="num">95% CI</th><th className="num">p</th><th className="num">W / L / T</th></tr></thead>
                <tbody>
                  {Object.entries(r.significance).map(([k, v]: [string, Any]) => v["nDCG@5"] && (
                    <tr key={k} className={k.startsWith("B7_adaptive_hybrid vs B6") ? "highlight" : ""}>
                      <td>{k.replace(/_/g, " ")}</td>
                      <td className="num">{v["nDCG@5"].mean_diff >= 0 ? "+" : ""}{fmt(v["nDCG@5"].mean_diff)}</td>
                      <td className="num">[{fmt(v["nDCG@5"].ci95[0])}, {fmt(v["nDCG@5"].ci95[1])}]</td>
                      <td className="num">{fmt(v["nDCG@5"].p_value)}</td>
                      <td className="num">{v["nDCG@5"].wins}/{v["nDCG@5"].losses}/{v["nDCG@5"].ties}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid-cols section">
            <div className="card">
              <h2>Ablation</h2>
              <MetricTable summary={r.summary} rows={[["B7_adaptive_hybrid", "Full system"], ...ABL]} />
            </div>
            <div className="card">
              <h2>nDCG@5 by intent</h2>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>System</th>{Object.keys(INTENT_LABEL).map((i) => <th key={i} className="num">{INTENT_LABEL[i]}</th>)}</tr></thead>
                  <tbody>
                    {Object.entries(r.by_intent).filter(([k]) => ["B1_bm25", "B3_doc2vec_chunked_max", "B4_bm25_legal", "B6_fixed_hybrid", "B7_adaptive_hybrid"].includes(k)).map(([k, v]: [string, Any]) => (
                      <tr key={k}><td>{MAIN.find(([m]) => m === k)?.[1]}</td>
                        {Object.keys(INTENT_LABEL).map((i) => <td key={i} className="num">{fmt(v[i]["nDCG@5"])}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      <div className="grid-cols section">
        {ic && (
          <div className="card">
            <h2>Intent classifier</h2>
            <p style={{ fontSize: 14 }}>Selected <code>{ic.selected.model}</code> (C={ic.selected.C}) by grouped 5-fold CV (macro-F1 {fmt(ic.selected.cv_macro_f1)}).</p>
            <table>
              <thead><tr><th>Held-out set</th><th className="num">Accuracy</th><th className="num">Macro-F1</th></tr></thead>
              <tbody>
                <tr><td>Intent queries ({ic.n_test}) — classifier</td><td className="num">{fmt(ic.test.accuracy)}</td><td className="num">{fmt(ic.test.macro_f1)}</td></tr>
                <tr><td>Intent queries — keyword rules</td><td className="num">{fmt(ic.rule_baseline_test.accuracy)}</td><td className="num">{fmt(ic.rule_baseline_test.macro_f1)}</td></tr>
                {r && <tr><td>Retrieval test queries ({r.intent_on_retrieval_queries.n})</td><td className="num">{fmt(r.intent_on_retrieval_queries.classifier.accuracy)}</td><td className="num">{fmt(r.intent_on_retrieval_queries.classifier.macro_f1)}</td></tr>}
              </tbody>
            </table>
          </div>
        )}
        {ex && (
          <div className="card">
            <h2>Legal extraction quality</h2>
            <table>
              <thead><tr><th>Extractor</th><th className="num">P</th><th className="num">R</th><th className="num">F1</th><th className="num">n</th></tr></thead>
              <tbody>
                {Object.entries(ex.results).map(([k, v]: [string, Any]) => (
                  <tr key={k}><td>{v.label}</td><td className="num">{fmt(v.precision)}</td><td className="num">{fmt(v.recall)}</td><td className="num">{fmt(v.f1)}</td><td className="num">{v.n_docs}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="faint" style={{ marginTop: 8 }}>{ex.note}</p>
          </div>
        )}
        {r && (
          <div className="card">
            <h2>Efficiency</h2>
            <table>
              <tbody>
                {Object.entries(r.efficiency.latency_ms_median).map(([k, v]) => (
                  <tr key={k}><td>{k.replace("_ms", "")}</td><td className="num">{String(v)} ms</td><td className="num faint">p95 {r.efficiency.latency_ms_p95[k]} ms</td></tr>
                ))}
                <tr><td>Engine load</td><td className="num">{r.efficiency.engine_load_seconds} s</td><td /></tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {data.figures.length > 0 && (
        <div className="section">
          <h2>Figures</h2>
          <div className="grid-cols">
            {data.figures.map((f: string) => (
              <a key={f} className="card figure" href={`/api/evaluation/figures/${f}`} target="_blank" rel="noreferrer">
                <img src={`/api/evaluation/figures/${f}?t=${r?.generated ?? ""}`} alt={f} loading="lazy" />
                <div className="faint" style={{ marginTop: 6 }}>{f}</div>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
