import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { DashboardState } from '../types'
import { THEME } from '../theme'
import NetworkGraph from '../components/NetworkGraph'
import ScoreTimeline from '../components/ScoreTimeline'

const ATTACKS = [
  { label: 'DDoS', type: 'ddos' },
  { label: 'Port Scan', type: 'portscan' },
  { label: 'Lateral', type: 'lateral' },
  { label: 'Exfil', type: 'exfil' },
  { label: 'Web', type: 'web' },
]

const REFRESH_MS = 1500

export default function DashboardPage() {
  const [state, setState] = useState<DashboardState | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const [customScript, setCustomScript] = useState('')
  const [targetNode, setTargetNode] = useState('')
  const [injectStatus, setInjectStatus] = useState('')

  const refresh = useCallback(async () => {
    try {
      const s = await api.getState()
      setState(s)
    } catch {
      /* API may be starting */
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_MS)
    return () => clearInterval(id)
  }, [refresh])

  const showToast = (msg: string) => {
    setToast(msg)
    setTimeout(() => setToast(null), 3500)
  }

  const runAction = async (fn: () => Promise<{ state: DashboardState }>, successMsg?: string) => {
    setBusy(true)
    try {
      const res = await fn()
      setState(res.state)
      if (successMsg) showToast(successMsg)
    } catch (e) {
      showToast(e instanceof Error ? e.message : 'Action failed')
    } finally {
      setBusy(false)
    }
  }

  const handleAttack = (type: string, label: string) =>
    runAction(() => api.injectAttack(type), `💥 ${label} injected!`)

  const handleCustomInject = async () => {
    if (!targetNode) {
      setInjectStatus('⚠ Please select a target node.')
      return
    }
    if (!customScript.trim()) {
      setInjectStatus('⚠ Script cannot be empty.')
      return
    }
    setBusy(true)
    setInjectStatus('Submitting…')
    try {
      const res = await api.injectCustom(customScript, targetNode)
      setState(res.state)
      setInjectStatus(`⚡ Script injected → ${targetNode} (MSE: ${res.mse?.toFixed(4) ?? '—'})`)
      setTimeout(() => setInjectStatus(''), 3000)
    } catch (e) {
      setInjectStatus(`✗ ${e instanceof Error ? e.message : 'Failed'}`)
    } finally {
      setBusy(false)
    }
  }

  if (loading && !state) {
    return <div style={{ padding: '2rem', color: THEME.dim }}>Loading AURA dashboard…</div>
  }

  if (!state) {
    return (
      <div className="panel" style={{ margin: '2rem', color: THEME.red }}>
        API unreachable. Start backend: <code>python api_server.py</code>
      </div>
    )
  }

  const statusColor =
    state.system_status === 'ACTIVE'
      ? THEME.green
      : state.system_status === 'UNDER ATTACK'
        ? THEME.red
        : THEME.yellow

  const expl = state.last_explanation as Record<string, unknown> | null

  return (
    <>
      {/* Header */}
      <div
        className="panel"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
        }}
      >
        <div>
          <span style={{ fontSize: '1.4em', fontWeight: 'bold', color: THEME.cyan }}>🛡️ AURA</span>
          <span style={{ color: THEME.dim, marginLeft: '0.5em', fontSize: '0.85em' }}>
            Autonomous Unified Resilience Architecture
          </span>
          {state.org && (
            <span
              style={{
                marginLeft: '0.8em',
                background: `${state.org.color}22`,
                border: `1px solid ${state.org.color}`,
                borderRadius: 20,
                padding: '2px 12px',
                fontSize: '0.82em',
                color: state.org.color,
                fontWeight: 'bold',
              }}
            >
              {state.org.icon} {state.org.label.toUpperCase()} · {state.org.net}
            </span>
          )}
        </div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ color: statusColor, fontWeight: 'bold' }}>● {state.system_status}</span>
          <span style={{ color: THEME.dim, marginLeft: '1em', fontSize: '0.75em' }}>
            {state.model_status} | Blockchain: {state.blockchain_mode}
          </span>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid-6" style={{ marginBottom: '1rem' }}>
        {[
          ['Windows Processed', state.metrics.window_counter],
          ['Threats Detected', state.metrics.total_attacks],
          ['Nodes Blocked', state.metrics.total_blocked],
          ['FL Rounds', state.metrics.fl_rounds_done],
          ['Chain Entries', state.metrics.chain_entries],
          ['Current AE Score', state.metrics.current_ae_score.toFixed(4)],
        ].map(([label, val]) => (
          <div key={String(label)} className="metric-card">
            <div className="metric-label">{label}</div>
            <div className="metric-value">{val}</div>
          </div>
        ))}
      </div>

      {/* Graph + Timeline */}
      <div className="grid-2">
        <div>
          <h4 className="panel-title" style={{ color: THEME.cyan }}>🌐 Live Network Topology</h4>
          <NetworkGraph nodes={state.nodes} edgeIndex={state.edge_index} />
          <div style={{ fontSize: '0.75em', color: THEME.dim, marginTop: '0.25rem' }}>
            <span style={{ color: THEME.green }}>◆ Normal</span> &nbsp;
            <span style={{ color: THEME.yellow }}>◆ Evaluating</span> &nbsp;
            <span style={{ color: THEME.red }}>◆ Threat Detected</span> &nbsp;
            <span style={{ color: THEME.text }}>◇ Critical Infrastructure</span>
          </div>
        </div>
        <div>
          <h4 className="panel-title" style={{ color: THEME.cyan }}>📈 Anomaly Score Timeline</h4>
          <ScoreTimeline scores={state.timeline.scores} thresholds={state.timeline.thresholds} />
          {state.ema.warmup_left > 0 ? (
            <div style={{ fontSize: '0.8rem', color: THEME.blue, marginTop: '0.5rem' }}>
              🔄 EMA calibrating… {state.ema.warmup_left} windows remaining
            </div>
          ) : (
            <div style={{ fontSize: '0.75rem', color: THEME.dim, marginTop: '0.5rem' }}>
              EMA μ={state.ema.mean.toFixed(4)} σ={state.ema.std.toFixed(4)} threshold=
              {state.ema.threshold.toFixed(4)}
            </div>
          )}
        </div>
      </div>

      {/* Explanation panel */}
      {expl && (
        <>
          <hr className="divider" />
          <div className="panel">
            <h4 className="panel-title" style={{ color: THEME.cyan }}>🔍 AE Explainability Panel</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <p style={{ color: THEME.orange, fontWeight: 600 }}>
                  Inferred: {String(expl.inferred_attack)} ({Number(expl.match_score ?? 0).toFixed(1)}% match)
                </p>
                <p style={{ fontSize: '0.85rem', color: THEME.dim, marginTop: '0.5rem' }}>
                  Severity: <span className={`sev-${expl.severity}`}>{String(expl.severity)}</span>
                  {' · '}Confidence: {Number(expl.confidence ?? 0).toFixed(1)}%
                </p>
              </div>
              <div>
                <p style={{ fontSize: '0.8rem', color: THEME.dim, marginBottom: '0.35rem' }}>Top anomalous features:</p>
                {(expl.top_features as [string, number, number][] | undefined)?.slice(0, 5).map(([name, val], i) => (
                  <div key={i} style={{ fontSize: '0.78rem', fontFamily: 'monospace' }}>
                    {name}: {Number(val).toFixed(4)}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      <hr className="divider" />

      {/* Control panels */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 0.8fr', gap: '1rem' }}>
        {/* Attack */}
        <div className="panel">
          <h4 className="panel-title" style={{ color: THEME.red }}>🔴 Attack Simulation</h4>
          <div className="grid-3">
            {ATTACKS.map((a) => (
              <button
                key={a.type}
                className="btn btn-danger"
                disabled={busy}
                onClick={() => handleAttack(a.type, a.label)}
              >
                {a.label}
              </button>
            ))}
          </div>
          <div style={{ marginTop: '0.75rem' }}>
            <label style={{ fontSize: '0.72em', color: THEME.dim, textTransform: 'uppercase' }}>
              ⚡ Custom Script Injection
            </label>
            <textarea
              className="custom-script"
              placeholder="# Write your attack script here"
              value={customScript}
              onChange={(e) => setCustomScript(e.target.value)}
            />
            <select className="node-select" value={targetNode} onChange={(e) => setTargetNode(e.target.value)}>
              <option value="">Select target node</option>
              {state.nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.id} — {n.label}
                  {n.critical ? ' 🔑' : ''}
                </option>
              ))}
            </select>
            <button className="btn btn-amber" style={{ width: '100%', marginTop: '0.5rem' }} disabled={busy} onClick={handleCustomInject}>
              ⚡ Inject Custom Script
            </button>
            {injectStatus && <div style={{ marginTop: '0.4rem', fontSize: '0.75rem' }}>{injectStatus}</div>}
          </div>
          <button
            className="btn"
            style={{ width: '100%', marginTop: '0.75rem', borderColor: THEME.green, color: THEME.green }}
            disabled={busy}
            onClick={() => runAction(() => api.injectNormal(), '✅ Normal traffic processed')}
          >
            🟢 Generate Normal Traffic
          </button>
        </div>

        {/* Federation */}
        <div className="panel">
          <h4 className="panel-title" style={{ color: THEME.blue }}>🌐 Federation</h4>
          {state.org && (
            <div style={{ fontSize: '0.8rem', marginBottom: '0.5rem', color: state.under_attack ? THEME.red : THEME.dim }}>
              {state.under_attack ? '🚨 UNDER ATTACK — Quarantined' : state.fl_ready ? '✅ Ready for FL' : '⏸ Not ready'}
            </div>
          )}
          {state.org && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginBottom: '0.75rem' }}>
              {state.under_attack ? (
                <button className="btn btn-primary" disabled={busy} onClick={() => runAction(() => api.resolveAttack())}>
                  ✅ Issue Resolved — Resume
                </button>
              ) : (
                <>
                  <button
                    className="btn"
                    disabled={busy}
                    onClick={() => runAction(() => api.setFlReady(!state.fl_ready))}
                  >
                    {state.fl_ready ? '⏸ Revoke FL Readiness' : '✅ Signal FL Ready'}
                  </button>
                  <button className="btn btn-danger" disabled={busy} onClick={() => runAction(() => api.setUnderAttack())}>
                    🚨 Report Under Attack — Quarantine
                  </button>
                </>
              )}
            </div>
          )}
          <button
            className="btn btn-primary"
            style={{ width: '100%' }}
            disabled={busy || state.fl_running}
            onClick={() => runAction(() => api.runFederation(), 'Federation complete')}
          >
            {state.fl_running ? '⏳ Running FL…' : '🚀 Run FL Simulation'}
          </button>
          {state.fed_log.length > 0 && (
            <div className="log-list" style={{ marginTop: '0.75rem', color: THEME.blue }}>
              {state.fed_log.map((line, i) => (
                <div key={i} className="log-item">{line}</div>
              ))}
            </div>
          )}
        </div>

        {/* Blockchain */}
        <div className="panel">
          <h4 className="panel-title" style={{ color: THEME.cyan }}>⛓ Blockchain Audit</h4>
          <button
            className="btn"
            style={{ width: '100%', marginBottom: '0.5rem' }}
            disabled={busy}
            onClick={() => runAction(() => api.registerHash(), 'Hash registered')}
          >
            📝 Register Test Hash
          </button>
          <button
            className="btn btn-primary"
            style={{ width: '100%', marginBottom: '0.75rem' }}
            disabled={busy}
            onClick={async () => {
              try {
                const r = await api.verifyChain()
                showToast(r.ok ? '✅ Chain INTACT' : '⚠ TAMPER DETECTED')
              } catch (e) {
                showToast(e instanceof Error ? e.message : 'Verify failed')
              }
            }}
          >
            🔍 Verify Chain
          </button>
          {state.chain_log.map((e, i) => (
            <div key={i} className="log-item" style={{ fontFamily: 'monospace', fontSize: '0.74rem', color: THEME.cyan }}>
              R{e.round} {e.version}: {e.hash.slice(0, 20)}… @ {e.time}
            </div>
          ))}
        </div>
      </div>

      <hr className="divider" />

      {/* Logs */}
      <div className="grid-2">
        <div className="panel">
          <h4 className="panel-title" style={{ color: THEME.yellow }}>🔔 Alert History</h4>
          {state.alerts.length === 0 ? (
            <span style={{ color: THEME.dim }}>No alerts triggered yet.</span>
          ) : (
            <div className="log-list">
              {state.alerts.map((a, i) => (
                <div key={i} className="log-item">
                  <span className={`sev-${a.severity}`}>{String(a.severity)}</span>
                  {' · '}
                  {String(a.inferred_attack ?? a.tag ?? 'Alert')}
                  {' · MSE '}
                  {Number(a.ae_score ?? 0).toFixed(4)}
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="panel">
          <h4 className="panel-title" style={{ color: THEME.orange }}>🛡️ Response Actions</h4>
          {state.incidents.length === 0 ? (
            <span style={{ color: THEME.dim }}>No responses triggered yet.</span>
          ) : (
            <div className="log-list">
              {state.incidents.map((r, i) => (
                <div key={i} className="log-item">
                  <strong>{String(r.action_taken)}</strong> → {String(r.node_id)} ({String(r.node_label)})
                  <br />
                  <span style={{ color: THEME.dim, fontSize: '0.72rem' }}>{String(r.policy_reason)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: '1rem', textAlign: 'right' }}>
        <button className="btn" disabled={busy} onClick={() => runAction(() => api.clearLogs(), 'Logs cleared')}>
          🗑️ Clear All Logs
        </button>
      </div>

      {toast && <div className="toast">{toast}</div>}
    </>
  )
}
