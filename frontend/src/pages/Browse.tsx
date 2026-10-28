import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

interface Row { case_id: string; case_name: string; decision_date: string; result: string; keywords: string }

export default function BrowsePage() {
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => {
      api.cases(q, offset).then((d) => { setRows(d.items); setTotal(d.total); });
    }, 200);
    return () => clearTimeout(t);
  }, [q, offset]);

  return (
    <div>
      <h1>Judgments</h1>
      <p className="muted">The corpus: Supreme Court of India judgments reported in 2024 (Digital SCR), from the public AWS Open Data collection.</p>
      <div className="row" style={{ margin: "16px 0" }}>
        <input type="search" placeholder="Filter by party name, citation or keyword" value={q}
          onChange={(e) => { setQ(e.target.value); setOffset(0); }} style={{ minWidth: 320, flex: 1, maxWidth: 520 }} />
        <span className="faint">{total} judgments</span>
      </div>
      <div className="card" style={{ padding: 0 }}>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Case</th><th>Date</th><th>Result</th><th>Keywords</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.case_id}>
                  <td><Link to={`/case/${encodeURIComponent(r.case_id)}`}>{r.case_name}</Link><div className="faint">{r.case_id}</div></td>
                  <td style={{ whiteSpace: "nowrap" }}>{r.decision_date}</td>
                  <td>{r.result}</td>
                  <td className="faint">{r.keywords}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <button className="btn small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous</button>
        <span className="faint">{offset + 1}–{Math.min(offset + 50, total)} of {total}</span>
        <button className="btn small" disabled={offset + 50 >= total} onClick={() => setOffset(offset + 50)}>Next</button>
      </div>
    </div>
  );
}
