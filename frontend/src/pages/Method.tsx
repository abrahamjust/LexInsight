import { useEffect, useState } from "react";
import { api, INTENT_LABEL, SIGNALS, SIGNAL_LABEL, type Meta } from "../api";
import { signalColor } from "../components/Signals";

const STEPS = [
  ["Corpus", "781 Supreme Court judgments (2024, Digital SCR) parsed from PDF into Issue, Headnotes, Case Law Cited, List of Acts, Keywords, judgment body and Result, with running page headers removed."],
  ["Query understanding", "Acts and provisions are extracted with regular expressions and an alias dictionary (IPC, CrPC, NDPS …; BNS/BNSS kept separate from IPC/CrPC). Legal-issue concepts, named precedents and requested outcomes are detected. A linear SVM over word and character TF-IDF plus legal-reference features predicts one of five intents."],
  ["Lexical evidence (L)", "BM25 Okapi over the cleaned judgment text."],
  ["Semantic evidence (S)", "Doc2Vec PV-DM, whole-document and 400-token chunks with 100-token overlap; query inference is seeded so rankings are reproducible."],
  ["Legal evidence (G)", "G = 0.30·Act match + 0.40·provision match + 0.30·legal-issue similarity, re-normalised over the parts the query actually uses. A provision found in the issue or headnotes counts as substantive (1.0); body mentions count 0.6 (≥3) or 0.3."],
  ["Citation evidence (C)", "Treatments from Case Law Cited are mapped to polarity (relied on/followed → positive; distinguished/overruled/held inapplicable → negative; referred to → neutral). A named precedent is matched directly, weighted by polarity for the precedent intents; otherwise authorities shared with the top seed results (bibliographic coupling) are scored."],
  ["Outcome evidence (O)", "Procedural result from the editor's Result line; substantive outcome (acquittal, conviction upheld, bail granted, quashed …) from cue phrases in the operative paragraphs. O = 1 compatible, 0 conflicting, 0.5 unknown; only active when the query asks for an outcome."],
  ["Fusion", "Signals are min–max normalised over the candidate pool (top 100 of each component). H = Σ w_j(intent)·signal_j, with weights for inapplicable signals redistributed."],
];

export default function MethodPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  useEffect(() => { api.meta().then(setMeta).catch(() => undefined); }, []);
  return (
    <div style={{ maxWidth: 1000 }}>
      <h1>Method</h1>
      <p className="muted">LexInsight retrieves and ranks existing judgments; it does not generate legal advice, and its explanations are templates filled from extracted features. A highly ranked judgment may still be distinguishable or superseded, so results need verification by a lawyer.</p>
      <div className="section">
        {STEPS.map(([t, d], i) => (
          <div className="card" key={t}>
            <h3>{i + 1}. {t}</h3>
            <p className="muted" style={{ marginBottom: 0, fontSize: 14.5 }}>{d}</p>
          </div>
        ))}
      </div>
      {meta && (
        <div className="card section">
          <h2>Intent-specific weights in use</h2>
          <p className="faint">Configuration <code>{meta.defaults.adaptive}</code>, selected on the validation queries.</p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Intent</th>{SIGNALS.map((s) => <th key={s} className="num"><i style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: signalColor(s), marginRight: 5 }} />{SIGNAL_LABEL[s]}</th>)}</tr></thead>
              <tbody>
                {Object.entries(meta.adaptive_weights).map(([intent, row]) => (
                  <tr key={intent}><td>{INTENT_LABEL[intent]}</td>{SIGNALS.map((s) => <td key={s} className="num">{row[s].toFixed(3)}</td>)}</tr>
                ))}
                <tr className="highlight"><td>Fixed-weight hybrid (B6)</td>{SIGNALS.map((s) => <td key={s} className="num">{meta.fixed_weights[s].toFixed(3)}</td>)}</tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
