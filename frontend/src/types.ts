export interface OrgProfile {
  label: string
  id: string
  net: string
  icon: string
  role: string
  color: string
}

export interface NodeInfo {
  id: string
  label: string
  index: number
  critical: boolean
  color: string
  state: string
}

export interface DashboardState {
  system_status: string
  model_status: string
  blockchain_mode: string
  org: OrgProfile | null
  metrics: {
    window_counter: number
    total_attacks: number
    total_blocked: number
    fl_rounds_done: number
    chain_entries: number
    current_ae_score: number
  }
  nodes: NodeInfo[]
  edge_index: number[][] | null
  timeline: {
    scores: number[]
    thresholds: number[]
    timestamps: number[]
  }
  last_explanation: Record<string, unknown> | null
  alerts: Record<string, unknown>[]
  incidents: Record<string, unknown>[]
  fed_log: string[]
  chain_log: { version: string; hash: string; round: string | number; time: string }[]
  fl_client_status: Record<string, unknown>[]
  fl_ready: boolean
  under_attack: boolean
  fl_running: boolean
  ema: {
    warmup_left: number
    mean: number
    std: number
    threshold: number
  }
  critical_allowlist: Record<string, string>
}

export interface FLServerState {
  run_state: string
  fl_running: boolean
  fl_done: boolean
  current_round: number
  total_rounds: number
  pipe_steps: { icon: string; label: string; state: number }[]
  orgs: {
    id: string
    key: string
    label: string
    net: string
    icon: string
    ready: boolean
    under_attack: boolean
    net_live: string
  }[]
  client_cards: Record<string, { status: string; round: number; selected: boolean | null; verified: boolean | null }>
  round_results: Record<string, unknown>[]
  hash_ledger: { version: string; hash: string; round: number; time: string }[]
  fl_log: string[]
  fltrust_scores_hist: { round: number; scores: Record<string, number> }[]
  global_hash: string | null
  global_version: string | null
  byzantine_org: string | null
  quarantined_orgs: string[]
}
