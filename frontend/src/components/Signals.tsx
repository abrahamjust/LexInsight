import { SIGNALS, SIGNAL_LABEL, type Signal } from "../api";

export const signalColor = (s: Signal) => `var(--s-${s})`;

export function SignalLegend() {
  return (
    <div className="legend" aria-label="Signal colours">
      {SIGNALS.map((s) => (
        <span key={s}>
          <i style={{ background: signalColor(s) }} />
          {SIGNAL_LABEL[s]}
        </span>
      ))}
    </div>
  );
}

/** Stacked bar of each signal's weighted contribution to the hybrid score. */
export function ContributionBar({ contributions, total }: { contributions: Partial<Record<Signal, number>>; total: number }) {
  const parts = SIGNALS.filter((s) => (contributions[s] ?? 0) > 0);
  const denom = Math.max(total, 1e-9);
  return (
    <div>
      <div className="stack" role="img"
        aria-label={parts.map((s) => `${SIGNAL_LABEL[s]} ${(contributions[s] ?? 0).toFixed(3)}`).join(", ")}>
        {parts.map((s) => (
          <span key={s} title={`${SIGNAL_LABEL[s]}: ${(contributions[s] ?? 0).toFixed(3)}`}
            style={{ width: `${((contributions[s] ?? 0) / denom) * 100}%`, background: signalColor(s) }} />
        ))}
      </div>
    </div>
  );
}

export function Meter({ label, value, color, off, suffix }: { label: string; value: number; color: string; off?: boolean; suffix?: string }) {
  return (
    <div className={`meter${off ? " off" : ""}`}>
      <span>{label}</span>
      <div className="track">
        <div className="fill" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} />
      </div>
      <span className="val">{off ? "n/a" : suffix ?? value.toFixed(2)}</span>
    </div>
  );
}
