import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { FLServerState } from '../types'
import { FL_THEME as T } from '../theme'

const REFRESH_MS = 1000

export default function FLServerPage() {
  const [state, setState] = useState<FLServerState | null>(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setState(await api.getFLState())
    } catch {
      /* API starting */
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])

  const runFL = async () => {
    setBusy(true)
    try {
      const res = await api.runFLSimulation()
      setState(res.state)
    } finally {
      setBusy(false)
    }
  }

  if (!state) {
    return <div style={{ padding: '2rem', color: T.dim }}>Loading FL Server Console…</div>
  }

  const runColor = state.fl_running ? T.yellow : state.fl_done ? T.green : T.dim

  return (
    <>
      <div
        className="panel"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '0.8rem',
          background: T.panel,
          borderColor: T.border,
        }}
      >
        <div>
          <span style={{ fontSize: '1.5em', fontWeight: 'bold', color: T.cyan }}>⚙️ AURA · FL Server Console</span>
          <span style={{ color: T.dim, marginLeft: '0.8em', fontSize: '0.82em' }}>
            FLTrust-Aggregated Federated Learning · Blockchain-Audited
          </span>
        </div>
        <div>
          <span style={{ color: runColor, fontWeight: 'bold' }}>● {state.run_state}</span>
          <span style={{ color: T.dim, marginLeft: '1em', fontSize: '0.78em' }}>
            Round {state.current_round}/{state.total_rounds}
          </span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
        {[
          ['Rounds Done', `${state.current_round} / ${state.total_rounds}`],
          ['FLTrust trusted', state.fl_done ? '3 / 3' : '— / 3'],
          ['Global Version', state.global_version ?? '—'],
          ['Status', state.run_state],
        ].map(([label, val]) => (
          <div key={label} className="metric-card" style={{ background: T.panel, borderColor: T.border }}>
            <div className="metric-label" style={{ color: T.dim }}>{label}</div>
            <div className="metric-value" style={{ color: T.cyan, fontSize: '1.2rem' }}>{val}</div>
          </div>
        ))}
      </div>

      <hr className="divider" style={{ borderColor: T.border }} />

      <h4 style={{ color: T.green, marginBottom: '0.5rem' }}>📡 Org Node Readiness</h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
        {state.orgs.map((o) => {
          const border = o.under_attack ? T.red : o.ready ? T.green : T.border
          return (
            <div
              key={o.key}
              style={{
                background: T.panel,
                border: `1px solid ${border}`,
                borderRadius: 10,
                padding: '0.9rem 1.1rem',
              }}
            >
              <div style={{ fontWeight: 600 }}>
                {o.icon} {o.label}
              </div>
              <div style={{ fontSize: '0.78rem', color: T.dim, marginTop: '0.25rem' }}>{o.net_live}</div>
              <div style={{ fontSize: '0.8rem', marginTop: '0.35rem', color: o.under_attack ? T.red : o.ready ? T.green : T.dim }}>
                {o.under_attack ? '🚨 QUARANTINED' : o.ready ? '✅ READY' : '⏸ IDLE'}
              </div>
            </div>
          )
        })}
      </div>

      <h4 style={{ color: T.cyan, marginBottom: '0.5rem' }}>Pipeline</h4>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${state.pipe_steps.length}, 1fr)`, gap: '0.5rem', marginBottom: '1rem' }}>
        {state.pipe_steps.map((step, i) => {
          const cls =
            step.state === 2 ? { border: T.green, bg: '#0f2117' } :
            step.state === 1 ? { border: T.cyan, bg: '#1a2a38' } :
            { border: T.border, bg: T.panel2, opacity: 0.45 }
          return (
            <div
              key={i}
              style={{
                background: cls.bg,
                border: `1px solid ${cls.border}`,
                borderRadius: 8,
                padding: '0.5rem 0.7rem',
                textAlign: 'center',
                fontSize: '0.78em',
                opacity: cls.opacity ?? 1,
              }}
            >
              <div style={{ fontSize: '1.2em' }}>{step.icon}</div>
              <div style={{ whiteSpace: 'pre-line', marginTop: '0.25rem' }}>{step.label}</div>
            </div>
          )
        })}
      </div>

      <button
        className="btn btn-primary"
        style={{ width: '100%', marginBottom: '1rem', borderColor: T.cyan, color: T.cyan }}
        disabled={busy || state.fl_running}
        onClick={runFL}
      >
        {state.fl_running ? '⏳ FL Running…' : '🚀 Start FL Simulation'}
      </button>

      {state.global_hash && (
        <div className="panel" style={{ background: T.panel, borderColor: T.border, marginBottom: '1rem' }}>
          <div style={{ color: T.purple, fontSize: '0.78rem', fontFamily: 'monospace' }}>
            Global hash: {state.global_hash}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div className="panel" style={{ background: T.panel, borderColor: T.border }}>
          <h4 style={{ color: T.purple, fontSize: '0.85rem', marginBottom: '0.5rem' }}>⛓ Hash Ledger</h4>
          {state.hash_ledger.length === 0 ? (
            <span style={{ color: T.dim, fontSize: '0.8rem' }}>No hashes minted yet.</span>
          ) : (
            state.hash_ledger.map((e, i) => (
              <div
                key={i}
                style={{
                  borderLeft: `3px solid ${T.purple}`,
                  padding: '0.35rem 0.6rem',
                  margin: '0.25rem 0',
                  fontSize: '0.74em',
                  fontFamily: 'monospace',
                }}
              >
                R{e.round} {e.version}: {e.hash.slice(0, 28)}… @ {e.time}
              </div>
            ))
          )}
        </div>
        <div className="panel" style={{ background: T.panel, borderColor: T.border }}>
          <h4 style={{ color: T.dim, fontSize: '0.85rem', marginBottom: '0.5rem' }}>FL Log</h4>
          <div className="log-list" style={{ maxHeight: 240 }}>
            {state.fl_log.length === 0 ? (
              <span style={{ color: T.dim }}>Waiting for FL run…</span>
            ) : (
              state.fl_log.map((line, i) => (
                <div key={i} style={{ color: T.dim, fontSize: '0.75em', fontFamily: 'monospace', padding: '0.2rem 0' }}>
                  {line}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  )
}
