/* eslint-disable @typescript-eslint/no-explicit-any -- judgment payload is passed through from the API as-is */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";

const TABS = ["Summary", "Legal features", "Citations", "Full judgment"] as const;

type CaseData = any;

function treatmentClass(polarity: string) {
  return polarity === "positive" ? "good" : polarity === "negative" ? "bad" : "";
}

export default function CasePage() {
  const { id } = useParams();
  const [data, setData] = useState<CaseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Summary");

  useEffect(() => {
    setData(null);
    api.case(id!).then(setData).catch((e) => setError(e.message));
  }, [id]);

  if (error) return <div className="banner bad">{error}</div>;
  if (!data) return <div className="row"><span className="spinner" /> Loading judgment…</div>;

  const s = data.sections;
  const f = data.features;
  return (
    <div>
      <Link to={-1 as unknown as string} onClick={(e) => { e.preventDefault(); history.back(); }} className="faint">← Back</Link>
      <h1 style={{ marginTop: 8 }}>{data.case_name}</h1>
      <div className="row muted" style={{ fontSize: 14, marginBottom: 16 }}>
        <span>{data.neutral_citation}</span>
        {data.scr_citation && <span>· {data.scr_citation}</span>}
        <span>· {data.decision_date}</span>
        {data.case_number && <span>· {data.case_number}</span>}
        {data.judges?.length > 0 && <span>· {data.judges.join(", ")}{data.author ? ` (author: ${data.author})` : ""}</span>}
        <a className="btn small" href={`/api/cases/${encodeURIComponent(data.case_id)}/pdf`} target="_blank" rel="noreferrer">Open PDF</a>
      </div>

      <div className="tabs" role="tablist">
        {TABS.map((t) => <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)} role="tab">{t}</button>)}
      </div>

      {tab === "Summary" && (
        <div className="grid-2">
          <div>
            <div className="card">
              <div className="card-title">Result</div>
              <div style={{ fontWeight: 600 }}>{s.result || "—"}</div>
              <div className="faint">Disposal (metadata): {data.disposal_nature ?? "—"}</div>
            </div>
            <div className="card">
              <div className="card-title">Acts</div>
              <p style={{ fontSize: 14 }}>{s.acts || "—"}</p>
            </div>
            <div className="card">
              <div className="card-title">Keywords</div>
              <p style={{ fontSize: 14 }}>{s.keywords || "—"}</p>
            </div>
          </div>
          <div>
            <div className="card">
              <div className="card-title">Issue for consideration</div>
              <p className="prose">{s.issue || "—"}</p>
            </div>
            <div className="card">
              <div className="card-title">Headnotes</div>
              <p className="prose">{s.headnotes || "—"}</p>
            </div>
          </div>
        </div>
      )}

      {tab === "Legal features" && (
        <div className="grid-cols">
          <div className="card">
            <div className="card-title">Acts (extracted)</div>
            <table>
              <thead><tr><th>Act</th><th className="num">Head</th><th className="num">Body</th></tr></thead>
              <tbody>
                {Object.entries(f.acts).sort((a: [string, any], b: [string, any]) => (b[1].head + b[1].body) - (a[1].head + a[1].body)).slice(0, 15).map(([k, v]: [string, any]) => (
                  <tr key={k}><td>{v.name}{v.year ? `, ${v.year}` : ""} {v.listed && <span className="chip good">listed</span>}</td>
                    <td className="num">{v.head}</td><td className="num">{v.body}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <div className="card-title">Provisions (most salient)</div>
            <table>
              <thead><tr><th>Provision</th><th className="num">Head</th><th className="num">Body</th></tr></thead>
              <tbody>
                {Object.entries(f.provisions).slice(0, 20).map(([k, v]: [string, any]) => (
                  <tr key={k}><td>{v.label}</td><td className="num">{v.head}</td><td className="num">{v.body}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="faint" style={{ marginTop: 8 }}>Head = issue, headnotes, acts and keywords. A provision present there is treated as substantively discussed; body-only mentions count less.</p>
          </div>
          <div className="card">
            <div className="card-title">Legal issues</div>
            <div className="chips">
              {Object.entries(f.concepts).map(([k, v]: [string, any]) => (
                <span key={k} className={`chip ${v.head ? "accent" : ""}`}>{k}</span>
              ))}
            </div>
            <div className="card-title" style={{ marginTop: 18 }}>Outcome</div>
            <p style={{ fontSize: 14 }}>Procedural: <b>{f.outcome.procedural ?? "unknown"}</b></p>
            <div className="chips">
              {Object.entries(f.outcome.substantive).map(([k, v]: [string, any]) => (
                <span key={k} className="chip warn">{k.replace(/_/g, " ")} ({Number(v).toFixed(1)})</span>
              ))}
              {Object.keys(f.outcome.substantive).length === 0 && <span className="muted" style={{ fontSize: 13.5 }}>No substantive outcome cue found</span>}
            </div>
          </div>
        </div>
      )}

      {tab === "Citations" && (
        <div className="card">
          <div className="card-title">Case law cited ({f.citations.length})</div>
          {(f.cites_in_corpus.length > 0 || f.cited_by_in_corpus.length > 0) && (
            <p style={{ fontSize: 14 }}>
              In this corpus — cites: {f.cites_in_corpus.map((c: string) => <Link key={c} to={`/case/${c}`} style={{ marginRight: 8 }}>{c}</Link>)}
              {f.cites_in_corpus.length === 0 && "none"}; cited by: {f.cited_by_in_corpus.map((c: string) => <Link key={c} to={`/case/${c}`} style={{ marginRight: 8 }}>{c}</Link>)}
              {f.cited_by_in_corpus.length === 0 && "none"}
            </p>
          )}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Authority</th><th>Citation</th><th>Treatment</th></tr></thead>
              <tbody>
                {f.citations.map((c: any, i: number) => (
                  <tr key={i}><td>{c.name}</td><td className="faint">{c.citations.join("; ")}</td>
                    <td><span className={`chip ${treatmentClass(c.polarity)}`}>{c.treatment}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === "Full judgment" && (
        <div className="card"><p className="prose">{s.judgment}</p></div>
      )}
    </div>
  );
}
