import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, INTENT_LABEL } from "../api";

interface QRow { qid: string; text: string; intent: string; split: string; candidates: number; drafted: number; reviewed: number; relevant: number }
interface Label { grade: number; reason?: string; note?: string }
interface Cand { case_id: string; case_name: string; issue: string; acts: string; keywords: string; result: string; cited: string[]; draft: Label | null; reviewed: Label | null }
interface QDetail { qid: string; text: string; intent: string; need: string; split: string; candidates: Cand[] }

const GRADE_TEXT = ["Not relevant", "Partially relevant", "Highly relevant"];

function GradeButtons({ value, onChange }: { value: number | null; onChange: (g: number) => void }) {
  return (
    <div className="grade" role="radiogroup" aria-label="Relevance grade">
      {[0, 1, 2].map((g) => (
        <button key={g} className={value === g ? `sel-${g}` : ""} onClick={() => onChange(g)} title={GRADE_TEXT[g]} role="radio" aria-checked={value === g}>
          {g}
        </button>
      ))}
    </div>
  );
}

export default function LabelsPage() {
  const [rows, setRows] = useState<QRow[]>([]);
  const [prov, setProv] = useState<Record<string, number> | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [detail, setDetail] = useState<QDetail | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadList = useCallback(() => {
    api.labels().then((d) => { setRows(d.queries); setProv(d.provenance); if (!active && d.queries.length) setActive(d.queries[0].qid); })
      .catch((e) => setError(e.message));
  }, [active]);

  useEffect(() => { loadList(); }, [loadList]);
  useEffect(() => { if (active) api.labelQuery(active).then(setDetail).catch((e) => setError(e.message)); }, [active]);

  async function setGrade(c: Cand, g: number) {
    if (!detail) return;
    setSaving(c.case_id);
    try {
      await api.setLabel(detail.qid, c.case_id, g, g !== c.draft?.grade ? "changed from draft" : "confirmed");
      setDetail({ ...detail, candidates: detail.candidates.map((x) => x.case_id === c.case_id ? { ...x, reviewed: { grade: g } } : x) });
      loadList();
    } finally {
      setSaving(null);
    }
  }

  async function confirmAll() {
    if (!detail) return;
    await api.confirmAll(detail.qid);
    api.labelQuery(detail.qid).then(setDetail);
    loadList();
  }

  return (
    <div>
      <h1>Relevance labels</h1>
      <p className="muted" style={{ maxWidth: "80ch" }}>
        Each query's candidates are the pooled top results of every system, shuffled and shown without system names.
        Claude drafted a grade and a reason for each pair; your grade overrides the draft. Grades: <b>2</b> directly
        addresses the information need, <b>1</b> partially or related, <b>0</b> not relevant. After reviewing, refresh
        the metrics on the Evaluation page.
      </p>
      {prov && (
        <div className="banner info" style={{ margin: "12px 0 20px" }}>
          {prov.reviewed_pairs} of {prov.draft_pairs} drafted pairs reviewed · {prov.grades_changed_by_reviewer} grades changed by the reviewer.
        </div>
      )}
      {error && <div className="banner bad">{error}</div>}
      {rows.length === 0 && !error && <div className="banner">No pools yet — run <code>python -m lexinsight.evaluation pool --split validation</code>.</div>}

      <div className="grid-2">
        <div className="card qlist" style={{ padding: 8 }}>
          {rows.map((r) => (
            <button key={r.qid} className={`qitem ${active === r.qid ? "active" : ""}`} onClick={() => setActive(r.qid)}>
              <div className="row" style={{ gap: 6 }}>
                <b style={{ fontSize: 13 }}>{r.qid}</b>
                <span className="chip" style={{ fontSize: 11 }}>{r.split}</span>
                <span className="faint" style={{ fontSize: 11.5 }}>{INTENT_LABEL[r.intent]}</span>
              </div>
              <div style={{ fontSize: 13, color: "var(--text-2)", marginTop: 2 }}>{r.text.length > 90 ? r.text.slice(0, 88) + "…" : r.text}</div>
              <div className="faint" style={{ fontSize: 11.5 }}>{r.reviewed}/{r.candidates} reviewed · {r.relevant} relevant</div>
              <div className="progress"><span style={{ width: `${(r.reviewed / Math.max(1, r.candidates)) * 100}%` }} /></div>
            </button>
          ))}
        </div>

        <div>
          {detail && (
            <>
              <div className="card">
                <div className="row">
                  <span className="chip accent">{INTENT_LABEL[detail.intent]}</span>
                  <span className="chip">{detail.split}</span>
                  <span className="spacer" />
                  <button className="btn small" onClick={confirmAll}>Confirm all remaining drafts</button>
                </div>
                <h2 style={{ marginTop: 10 }}>{detail.text}</h2>
                <p className="muted" style={{ fontSize: 14, marginBottom: 0 }}><b>Information need:</b> {detail.need}</p>
              </div>
              {detail.candidates.map((c) => {
                const current = c.reviewed?.grade ?? null;
                const changed = c.reviewed && c.draft && c.reviewed.grade !== c.draft.grade;
                return (
                  <div className="card" key={c.case_id}>
                    <div className="row" style={{ alignItems: "flex-start" }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <Link to={`/case/${encodeURIComponent(c.case_id)}`} target="_blank" className="result-title">{c.case_name}</Link>
                        <div className="result-meta">{c.case_id} · {c.result}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <GradeButtons value={current} onChange={(g) => setGrade(c, g)} />
                        <div className="faint" style={{ marginTop: 4 }}>
                          {saving === c.case_id ? "saving…" : current === null ? `draft: ${c.draft?.grade ?? "—"} (unreviewed)` : changed ? `changed from draft ${c.draft?.grade}` : "reviewed"}
                        </div>
                      </div>
                    </div>
                    <p className="issue" style={{ marginTop: 10 }}><b>Issue:</b> {c.issue}</p>
                    <p className="faint" style={{ marginBottom: 4 }}><b>Acts:</b> {c.acts || "—"}</p>
                    <p className="faint" style={{ marginBottom: 4 }}><b>Keywords:</b> {c.keywords || "—"}</p>
                    {c.cited.length > 0 && <p className="faint" style={{ marginBottom: 4 }}><b>Relevant citations:</b> {c.cited.join(" · ")}</p>}
                    {c.draft?.reason && (
                      <p style={{ fontSize: 13, marginTop: 8, marginBottom: 0 }}>
                        <span className="chip">Claude draft: {c.draft.grade}</span> <span className="muted">{c.draft.reason}</span>
                      </p>
                    )}
                  </div>
                );
              })}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
