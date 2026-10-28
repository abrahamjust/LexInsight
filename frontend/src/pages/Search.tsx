import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, INTENT_LABEL, SIGNALS, SIGNAL_LABEL, type Meta, type SearchResponse, type SearchResult } from "../api";
import { ContributionBar, Meter, SignalLegend, signalColor } from "../components/Signals";

const EXAMPLES = [
  "Can a conviction be based solely on a dying declaration without corroboration?",
  "Find Supreme Court judgments interpreting Section 302 IPC.",
  "Cases following Gian Singh v. State of Punjab on quashing after compromise",
  "Judgments that distinguished Umadevi to protect long-serving employees",
  "A young wife died of burns at her in-laws' house and her parents allege dowry harassment",
  "Bail under Section 45 PMLA where the trial is delayed",
];

function QueryPanel({ data }: { data: SearchResponse }) {
  const q = data.query;
  const scores = Object.entries(q.intent_scores).sort((a, b) => b[1] - a[1]);
  return (
    <aside>
      <div className="card">
        <div className="card-title">Predicted intent</div>
        <div style={{ fontSize: "1.15rem", fontWeight: 600 }}>{INTENT_LABEL[q.intent] ?? q.intent}</div>
        <div className="faint" style={{ marginBottom: 10 }}>
          {q.intent_source === "override" ? "set manually" : "linear SVM over TF-IDF + legal-reference features"}
        </div>
        {q.intent_source !== "override" &&
          scores.map(([k, v]) => <Meter key={k} label={INTENT_LABEL[k] ?? k} value={v} color="var(--accent-2)" />)}
      </div>

      <div className="card">
        <div className="card-title">Legal references detected</div>
        <Detail label="Acts" items={Object.values(q.acts)} empty="none named" />
        <Detail label="Provisions" items={q.provisions.map((p) => q.provision_labels[p] ?? p)}
          empty={q.implied_provisions.length ? `none cited (implied: ${q.implied_provisions.length})` : "none cited"} />
        <Detail label="Legal issues" items={q.concepts} empty="none recognised" />
        <Detail label="Named precedents" items={q.precedent_names} empty="none" />
        <Detail label="Requested outcome" items={q.outcome_requirements.map((o) => o.replace(/_/g, " "))} empty="none" />
      </div>

      {data.effective_weights && (
        <div className="card">
          <div className="card-title">Signal weights used</div>
          {SIGNALS.map((s) => (
            <Meter key={s} label={SIGNAL_LABEL[s]} value={data.effective_weights?.[s] ?? 0} color={signalColor(s)}
              off={!data.applicable[s]} />
          ))}
          <p className="faint" style={{ marginTop: 8, marginBottom: 0 }}>
            Signals with nothing to say for this query (no provision, no requested outcome, no citation evidence) are
            switched off and their weight is redistributed.
          </p>
        </div>
      )}
      <p className="faint">
        {data.results.length} results · {data.timings_ms.total_ms} ms · semantic: {data.semantic_variant}
        {data.weights_name ? ` · weights: ${data.weights_name}` : ""}
      </p>
    </aside>
  );
}

function Detail({ label, items, empty }: { label: string; items: string[]; empty: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div className="faint">{label}</div>
      {items.length ? (
        <div className="chips" style={{ marginTop: 4 }}>
          {items.map((i) => <span className="chip accent" key={i}>{i}</span>)}
        </div>
      ) : (
        <div className="muted" style={{ fontSize: 13.5 }}>{empty}</div>
      )}
    </div>
  );
}

function ResultCard({ r, hybrid }: { r: SearchResult; hybrid: boolean }) {
  const ex = r.explanation;
  const total = Object.values(ex.contributions).reduce((a, b) => a + (b ?? 0), 0);
  const sub = ex.outcome.substantive;
  return (
    <article className="card result">
      <div className="result-head">
        <div className="rank">{r.rank}</div>
        <div style={{ minWidth: 0 }}>
          <Link to={`/case/${encodeURIComponent(r.case_id)}`} className="result-title">{r.case_name}</Link>
          <div className="result-meta">
            {r.neutral_citation} {r.scr_citation ? `· ${r.scr_citation}` : ""} · {r.decision_date}
            {r.result ? ` · ${r.result}` : ""}
          </div>
        </div>
        <div className="score">
          <b>{Number.isFinite(r.score) ? r.score.toFixed(3) : "—"}</b>
          <div className="faint">{hybrid ? "hybrid score" : "score"}</div>
        </div>
      </div>
      {hybrid && total > 0 && (
        <div style={{ marginTop: 12 }}>
          <ContributionBar contributions={ex.contributions} total={total} />
        </div>
      )}
      <p className="explain"><strong>Why it ranks here:</strong> {ex.summary}.</p>
      <div className="chips">
        {ex.matched_provisions.map((p) => (
          <span key={p.provision} className={`chip ${p.implied ? "" : "good"}`}>{p.provision} · {p.where}</span>
        ))}
        {ex.matched_concepts.map((c) => <span key={c} className="chip">{c}</span>)}
        {ex.citation?.mode === "named" && (
          <span className={`chip ${ex.citation.treatment === "negative" ? "bad" : ex.citation.treatment === "positive" ? "good" : ""}`}>
            cites {ex.citation.authority?.slice(0, 40)} ({ex.citation.treatment})
          </span>
        )}
        {Object.keys(sub).map((k) => <span key={k} className="chip warn">{k.replace(/_/g, " ")}</span>)}
      </div>
      <blockquote className="passage">{r.passage}</blockquote>
      <details>
        <summary>Issue for consideration and raw signal values</summary>
        <p className="issue" style={{ marginTop: 8 }}>{r.issue}</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Signal</th><th className="num">Raw</th><th className="num">Normalised</th><th className="num">Contribution</th></tr>
            </thead>
            <tbody>
              {SIGNALS.map((s) => (
                <tr key={s}>
                  <td>{SIGNAL_LABEL[s]}</td>
                  <td className="num">{r.signals[s].toFixed(4)}</td>
                  <td className="num">{r.normalized ? r.normalized[s].toFixed(3) : "—"}</td>
                  <td className="num">{ex.contributions[s] !== undefined ? ex.contributions[s]!.toFixed(3) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </article>
  );
}

function ResultList({ data }: { data: SearchResponse }) {
  const hybrid = !!data.effective_weights;
  if (!data.results.length) return <div className="banner">No judgments matched this query.</div>;
  return (
    <div>
      {hybrid && <div style={{ marginBottom: 12 }}><SignalLegend /></div>}
      {data.results.map((r) => <ResultCard key={r.case_id} r={r} hybrid={hybrid} />)}
    </div>
  );
}

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [query, setQuery] = useState(params.get("q") ?? "");
  const [system, setSystem] = useState(params.get("system") ?? "adaptive_hybrid");
  const [intent, setIntent] = useState<string>("");
  const [compare, setCompare] = useState<string>("");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [other, setOther] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(`Backend not reachable: ${e.message}`));
  }, []);

  async function run(q = query) {
    if (q.trim().length < 2) return;
    setLoading(true);
    setError(null);
    setParams({ q, system });
    try {
      const [a, b] = await Promise.all([
        api.search({ query: q, system, top_k: 10, intent_override: intent || null }),
        compare ? api.search({ query: q, system: compare, top_k: 10, intent_override: intent || null }) : Promise.resolve(null),
      ]);
      setData(a);
      setOther(b);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (params.get("q")) run(params.get("q")!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const systems = meta?.systems ?? { adaptive_hybrid: "B7 Intent-adaptive hybrid" };

  return (
    <div>
      <h1>Find relevant precedent</h1>
      <p className="muted" style={{ maxWidth: "75ch" }}>
        Ask a legal question, describe facts, cite a provision, or name a precedent. LexInsight classifies what kind of
        authority you want and weighs lexical, semantic, statutory, citation and outcome evidence accordingly
        {meta ? ` across ${meta.n_judgments} Supreme Court judgments from 2024` : ""}.
      </p>

      <div className="card" style={{ marginTop: 16 }}>
        <form className="search-box" onSubmit={(e) => { e.preventDefault(); run(); }}>
          <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Can a conviction rest on circumstantial evidence when no one saw the crime?"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } }} aria-label="Legal query" />
          <button className="btn primary" type="submit" disabled={loading}>{loading ? <span className="spinner" /> : "Search"}</button>
        </form>
        <div className="row" style={{ marginTop: 12 }}>
          <label className="field">System
            <select value={system} onChange={(e) => setSystem(e.target.value)}>
              {Object.entries(systems).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <label className="field">Intent
            <select value={intent} onChange={(e) => setIntent(e.target.value)}>
              <option value="">Predict automatically</option>
              {Object.entries(INTENT_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <label className="field">Compare with
            <select value={compare} onChange={(e) => setCompare(e.target.value)}>
              <option value="">No comparison</option>
              {Object.entries(systems).filter(([k]) => k !== system).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
        </div>
        <div className="examples">
          {EXAMPLES.map((e) => (
            <button key={e} className="chip" onClick={() => { setQuery(e); run(e); }}>{e.length > 70 ? e.slice(0, 68) + "…" : e}</button>
          ))}
        </div>
      </div>

      {error && <div className="banner bad section">{error}</div>}

      {data && !other && (
        <div className="grid-2 section">
          <QueryPanel data={data} />
          <ResultList data={data} />
        </div>
      )}
      {data && other && (
        <div className="section">
          <div className="compare">
            {[data, other].map((d) => (
              <div key={d.system}>
                <h2>{systems[d.system] ?? d.system}</h2>
                <p className="faint">Intent: {INTENT_LABEL[d.query.intent]} · {d.timings_ms.total_ms} ms</p>
                <ResultList data={d} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
