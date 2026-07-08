# AURA — Session Code Changes Report
## Branch: `Watts-Strogatz-tests`
### Commits covered: `d1a9155` → `16f7843`

---

## Overview

Three commits were made in this session, driven by two objectives:

1. **Research-grade reproducibility** — every numeric parameter that controls how the
   system splits data, escalates severity, or evaluates responses must live in a single
   canonical location (`config.py`), not scattered as magic numbers across scripts.

2. **Section 3.5 evaluation gap** — the Three-Tier HITL Response Engine was a *design
   description* with no measured metrics. This session turns it into a *scientific
   contribution* by adding a full evaluation benchmark.

---

## Commit 1 — `d1a9155`
### `refactor: centralize split fractions and paths into config.py to remove hardcodings`

### Problem
The train/test split ratio (`0.20`) and the calibration fraction (`0.10`) were hardcoded
as numeric literals in **five separate scripts**. Changing the split ratio required
editing five files by hand, with no guarantee of consistency.

Similarly, `aura/split_manager.py` constructed the `splits/` directory path using a
relative `Path(__file__)` expression instead of the canonical `BASE_DIR` from `config.py`.

---

### Changes

#### [`config.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/config.py) — MODIFIED

**Added:**
```python
# Directory for persisting canonical train/test split indices
SPLITS_DIR = BASE_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Fraction of windows held out for the canonical test set
TEST_SPLIT_FRACTION = 0.20

# Fraction of train windows used for threshold calibration
CALIB_SPLIT_FRACTION = 0.10
```

**Impact:** `BASE_DIR / "splits"` ensures the splits directory is always co-located
with the project root, regardless of where Python is invoked from. `TEST_SPLIT_FRACTION`
and `CALIB_SPLIT_FRACTION` are now the **single source of truth** — change either once
and all scripts below update automatically.

---

#### [`aura/split_manager.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/split_manager.py) — MODIFIED

**Before:**
```python
_SPLIT_DIR  = Path(__file__).resolve().parent.parent / "splits"
_SPLIT_FILE = _SPLIT_DIR / "canonical_split.npz"

def get_canonical_split(
    all_windows,
    test_fraction: float = 0.20,
    calib_fraction: float = 0.10,
    ...
```

**After:**
```python
import config as cfg
_SPLIT_FILE = cfg.SPLITS_DIR / "canonical_split.npz"

def get_canonical_split(
    all_windows,
    test_fraction: float = cfg.TEST_SPLIT_FRACTION,
    calib_fraction: float = cfg.CALIB_SPLIT_FRACTION,
    ...
```

**Why:** The old path was fragile — if the module was imported from a different working
directory, `Path(__file__).resolve()` would still work but was silently independent of
the `config.py`-controlled project layout. Now both the save path and the function
defaults come from the same authority.

---

#### [`train.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/train.py) — MODIFIED

**Before:**
```python
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=0.20)
n_val = int(len(all_benign) * 0.20)
```

**After:**
```python
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=cfg.TEST_SPLIT_FRACTION)
n_val = int(len(all_benign) * cfg.TEST_SPLIT_FRACTION)
```

**Impact:** The val split inside `train.py` now mirrors the canonical test fraction
exactly. Previously, if you changed `0.20` in one place but forgot the other, the model
would train with an inconsistent val/test boundary.

---

#### [`calibrate_thresholds.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/calibrate_thresholds.py) — MODIFIED

```python
# Before
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=0.20)

# After
_, train_windows, _ = get_canonical_split(all_windows, test_fraction=cfg.TEST_SPLIT_FRACTION)
```

**Impact:** Threshold calibration is performed on train-windows only. The literal `0.20`
was inconsistent with `benchmark_ablation.py` — if one used a different split than the
other, the calibrated thresholds would be evaluated on data they were fitted on.

---

#### [`scripts/benchmark_ablation.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/benchmark_ablation.py) — MODIFIED

```python
# Before (function signature and argparse default)
def collect_test_windows(loader, scaler, test_fraction: float = 0.20):
parser.add_argument("--test-fraction", type=float, default=0.20)

# After
def collect_test_windows(loader, scaler, test_fraction: float = cfg.TEST_SPLIT_FRACTION):
parser.add_argument("--test-fraction", type=float, default=cfg.TEST_SPLIT_FRACTION)
```

---

#### [`scripts/ablation_sweep.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/ablation_sweep.py) — MODIFIED
#### [`aura_attacks/ablation_sweep.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura_attacks/ablation_sweep.py) — MODIFIED

Both: `default=0.20` → `default=cfg.TEST_SPLIT_FRACTION` in argparse.

---

## Commit 2 — `cce88a4`
### `feat(sec3.5): implement HITL response engine benchmark`

### Problem
Section 3.5 of the paper describes a Three-Tier HITL Response Engine but provides
**zero evaluation metrics**. The paper's own evaluation note states:

> *"Escalation latency, false escalation rate, and operator workload must be measured.
> Without these numbers the response layer is not a scientific contribution —
> it is a design description. Response engine evaluation belongs in Tier 2."*

---

### Changes

#### [`config.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/config.py) — MODIFIED

**Added — Section 3.5 HITL constants (all sourced from the paper's own table):**

```python
# "3 LOWs within 5-min window" -> MEDIUM (Section 3.5 trigger, row 2)
HITL_LOW_TO_MEDIUM_THRESHOLD  = 3

# "3 MEDIUMs within window" -> HIGH (Section 3.5 trigger, row 3, second condition)
HITL_MEDIUM_TO_HIGH_THRESHOLD = 3

# "5 LOWs within window" -> HIGH (Section 3.5 trigger, row 3, first condition)
HITL_LOW_TO_HIGH_THRESHOLD    = 5

# Seconds before HITL times out -> DEGRADED tier
HITL_TIMEOUT_SEC              = 30

# Simulated operator approval probability for benchmark only [0.0, 1.0]
# DEGRADED rate in benchmark = 1.0 - HITL_APPROVAL_RATE
HITL_APPROVAL_RATE            = 0.85

# Deduplication window (was hardcoded 30 in response_engine.py)
RESPONSE_DEDUP_WINDOW_SEC     = 30
```

**Why each is a constant and not a magic number:**
- `HITL_LOW_TO_MEDIUM_THRESHOLD`, `HITL_LOW_TO_HIGH_THRESHOLD`, `HITL_MEDIUM_TO_HIGH_THRESHOLD` —
  these are the exact values in the Section 3.5 table. They control both the live
  `_apply_temporal_escalation()` in `detector.py` AND `benchmark_hitl_response.py`.
  Changing them in one place now propagates everywhere.
- `HITL_APPROVAL_RATE` is clearly documented as benchmark-only — it is never used by
  the live response engine.
- `RESPONSE_DEDUP_WINDOW_SEC` eliminates the last magic `30` in `response_engine.py`.

---

#### [`aura/response_engine.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/response_engine.py) — MODIFIED

**Change 1 — Remove hardcoded dedup window:**
```python
# Before
self._dedup_window_sec = 30   # Suppress duplicate actions within 30s

# After
self._dedup_window_sec = cfg.RESPONSE_DEDUP_WINDOW_SEC
```

**Change 2 — Add `act_with_latency()` method:**
```python
def act_with_latency(self, event: AnomalyEvent) -> tuple:
    """
    Identical to act() but returns (records, latency_ms).
    Uses monotonic clock for accurate wall-clock timing.
    Used by benchmark_hitl_response.py to measure escalation latency.
    """
    t0 = time.monotonic()
    records = self.act(event)
    return records, (time.monotonic() - t0) * 1000.0
```

**Why monotonic clock:** `time.time()` can jump backward (NTP sync). `time.monotonic()`
is guaranteed to be non-decreasing — essential for accurate latency measurement.

---

#### [`scripts/benchmark_hitl_response.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/scripts/benchmark_hitl_response.py) — NEW (~400 lines)

**Architecture:**

```
test_windows  (from get_canonical_split — never train windows)
      |
      v
AURAInferenceEngine.process(graph, labels)
      |
      v
BenchmarkResponseEngine.act_benchmark(event)
  - Simulated HITL via HITLSimulator (seeded random draw)
  - No disk writes (suppressed to avoid polluting production logs)
  - Returns (records, latency_ms, hitl_fired, hitl_approved)
      |
      v
classify_response_tier()
  -> NONE | LOW | MEDIUM | HIGH | DEGRADED
      |
      v
EscalationEvent (per-window record)
  - true_attack_ratio, is_attack_window
  - response_tier, latency_ms
  - hitl_fired, hitl_approved
  - is_false_escalation
      |
      v
TierMetrics aggregation (per tier)
      |
      v
reports/hitl_benchmark_results.{json,csv}
```

**Key classes:**

| Class | Role |
|-------|------|
| `HITLSimulator` | Seeded `random.Random` replaces blocking `input()`. Approval rate = `cfg.HITL_APPROVAL_RATE`. |
| `BenchmarkResponseEngine` | Subclass of `AURAResponseEngine`. Overrides `_write_record()` (in-memory only) and `_send_hitl_alert()` (suppressed). Intercepts HIGH events to inject simulated HITL. |
| `classify_response_tier()` | Maps `IncidentRecord` list + HITL decision to `NONE/LOW/MEDIUM/HIGH/DEGRADED`. |
| `EscalationEvent` | Dataclass: one row per window in the CSV output. |
| `TierMetrics` | Accumulates latencies and false escalation counts per tier. Computes P50/P95/P99. |
| `ResponseBenchmark` | Orchestrator: loads test split, runs inference + response, aggregates, exports. |

**Three metrics produced:**

| Metric | Formula | Operational threshold |
|--------|---------|----------------------|
| Escalation Latency P95 | `percentile(all_latencies_ms, 95)` | < 100ms |
| False Escalation Rate | `escalations on benign / total escalations` | < 10% |
| HITL calls/hour | `hitl_calls / simulated_hours` | < 10/hr |

**Dataset usage (split_manager):**
```python
# ONLY test_windows are evaluated — training data never used for evaluation
_, _, test_windows = get_canonical_split(
    all_windows, test_fraction=cfg.TEST_SPLIT_FRACTION
)
```

**CLI options (all default to config.py values):**
```bash
python scripts/benchmark_hitl_response.py                     # full run
python scripts/benchmark_hitl_response.py --quick             # 10% data
python scripts/benchmark_hitl_response.py --hitl-rate 0.70    # sensitivity
python scripts/benchmark_hitl_response.py --seed 99           # different seed
python scripts/benchmark_hitl_response.py --test-fraction 0.3 # different split
```

---

## Commit 3 — `16f7843`
### `fix: replace hardcoded HITL thresholds in detector.py with cfg constants`

### Problem
The **live inference engine** (`aura/detector.py`) had the Section 3.5 table values
hardcoded in `_apply_temporal_escalation()`:

```python
# BEFORE — magic numbers in production logic
if low_count >= 5 or medium_count >= 3:
    candidate = AlertSeverity.HIGH
elif low_count >= 3:
    candidate = AlertSeverity.MEDIUM
```

This was a critical inconsistency: `config.py` now held `HITL_LOW_TO_HIGH_THRESHOLD = 5`
but the live engine still had `5` in its source. Changing the constant in `config.py`
would have had **no effect** on the actual escalation behaviour.

---

### Change

#### [`aura/detector.py`](file:///c:/Users/SURAJ/Desktop/ISFCR-1/AURA/aura/detector.py) — MODIFIED

```python
# AFTER — config-driven thresholds
if (
    low_count    >= cfg.HITL_LOW_TO_HIGH_THRESHOLD
    or medium_count >= cfg.HITL_MEDIUM_TO_HIGH_THRESHOLD
):
    candidate = AlertSeverity.HIGH
elif low_count >= cfg.HITL_LOW_TO_MEDIUM_THRESHOLD:
    candidate = AlertSeverity.MEDIUM
else:
    candidate = base_severity
```

**Impact:** The same constants now control:
1. The live inference engine (`aura/detector.py`)
2. The Section 3.5 benchmark (`scripts/benchmark_hitl_response.py`)
3. The docstring examples in `config.py`

Changing `HITL_LOW_TO_HIGH_THRESHOLD` once in `config.py` propagates to all three
without touching any logic file.

---

## Hardcode Audit — Final Status

### Changed files scanned

| File | Status | Notes |
|------|--------|-------|
| `config.py` | ✅ CLEAN | `0.20` and `0.10` **are** the definitions — correct |
| `train.py` | ✅ CLEAN | Uses `cfg.TEST_SPLIT_FRACTION` |
| `calibrate_thresholds.py` | ✅ CLEAN | Uses `cfg.TEST_SPLIT_FRACTION` |
| `aura/split_manager.py` | ✅ CLEAN | Uses `cfg.SPLITS_DIR`, `cfg.TEST_SPLIT_FRACTION`, `cfg.CALIB_SPLIT_FRACTION` |
| `aura/detector.py` | ✅ CLEAN | Uses `cfg.HITL_LOW_TO_*`, `cfg.HITL_MEDIUM_TO_*` |
| `aura/response_engine.py` | ✅ CLEAN | Uses `cfg.RESPONSE_DEDUP_WINDOW_SEC` |
| `scripts/benchmark_ablation.py` | ✅ CLEAN | Uses `cfg.TEST_SPLIT_FRACTION` |
| `scripts/ablation_sweep.py` | ✅ CLEAN | Uses `cfg.TEST_SPLIT_FRACTION` |
| `scripts/benchmark_hitl_response.py` | ✅ CLEAN | All defaults from `cfg.*` |
| `aura_attacks/ablation_sweep.py` | ✅ CLEAN | Uses `cfg.TEST_SPLIT_FRACTION` |

### What lives in `config.py` now

| Category | Constants |
|----------|-----------|
| Paths | `BASE_DIR`, `MODELS_DIR`, `LOGS_DIR`, `SPLITS_DIR`, `CONTRACTS_DIR` |
| Data split | `TEST_SPLIT_FRACTION`, `CALIB_SPLIT_FRACTION`, `DATA_LOAD_FRACTION` |
| HITL escalation | `HITL_LOW_TO_MEDIUM_THRESHOLD`, `HITL_LOW_TO_HIGH_THRESHOLD`, `HITL_MEDIUM_TO_HIGH_THRESHOLD`, `HITL_TIMEOUT_SEC` |
| HITL benchmark | `HITL_APPROVAL_RATE` |
| Response engine | `RESPONSE_DEDUP_WINDOW_SEC`, `CONFIDENCE_LOW_THRESHOLD`, `CONFIDENCE_MED_THRESHOLD` |
| EMA detector | `EMA_ALPHA`, `EMA_SIGMA_MULTIPLIER`, `EMA_WARMUP_BATCHES`, `K_CONSECUTIVE_READINGS`, `TEMPORAL_WINDOW_SECONDS` |

### Acceptable "magic number" exceptions

The following numeric literals remain in code and are **intentional** — they are not
policy values but mathematical or structural constants:

| Location | Value | Reason |
|----------|-------|--------|
| `np.percentile(..., 50/95/99)` | `50, 95, 99` | Standard statistical percentile values — not tunable policy |
| `_realistic_profile()` padding | `0.35` | Physics-derived benign baseline mean — not a threshold |
| `aura/models.py` layer widths | e.g. `128, 64` | Neural architecture — versioned via `AURAModelBundle` |

---

## Summary — What Changed and Why

```
config.py          [SPLITS_DIR + TEST/CALIB fractions]      → single source of truth for all splits
config.py          [HITL_* thresholds + APPROVAL_RATE]      → Section 3.5 table → code, no magic numbers
config.py          [RESPONSE_DEDUP_WINDOW_SEC]              → removes last magic 30s from response_engine
aura/split_manager → cfg.SPLITS_DIR, cfg.TEST/CALIB_*       → consistent path + default fractions
train.py           → cfg.TEST_SPLIT_FRACTION                → val/test boundary matches across all scripts
calibrate_thresholds → cfg.TEST_SPLIT_FRACTION              → calibration uses same split as evaluation
benchmark_ablation → cfg.TEST_SPLIT_FRACTION                → ablation and calibration use same partition
ablation_sweep (×2)→ cfg.TEST_SPLIT_FRACTION                → consistent CLI default
aura/detector.py   → cfg.HITL_LOW/MEDIUM/HIGH thresholds   → live engine uses config, not magic 3/5/3
aura/response_engine → cfg.RESPONSE_DEDUP_WINDOW_SEC        → no hardcoded 30s
                   → act_with_latency()                     → monotonic-clock timing for benchmark
scripts/benchmark_hitl_response.py [NEW]
   - HITLSimulator              seeded, reproducible operator simulation
   - BenchmarkResponseEngine    no disk I/O, no blocking stdin
   - classify_response_tier     NONE/LOW/MEDIUM/HIGH/DEGRADED mapping
   - ResponseBenchmark          test-split-only orchestrator
   - Metrics: FER, Latency P50/P95/P99, HITL calls/hour
   - Output: reports/hitl_benchmark_results.{json,csv}
```
