import Plot from 'react-plotly.js'
import { THEME } from '../theme'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

interface Props {
  scores: number[]
  thresholds: number[]
}

export default function ScoreTimeline({ scores, thresholds }: Props) {
  if (!scores.length) {
    return (
      <Plot
        data={[]}
        layout={{
          paper_bgcolor: THEME.bg,
          plot_bgcolor: THEME.bg,
          height: 200,
          annotations: [{ text: 'Awaiting data…', showarrow: false, font: { color: THEME.dim, size: 14 } }],
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%' }}
      />
    )
  }

  const t = scores.map((_, i) => i)
  const traces: Trace[] = [
    {
      x: t,
      y: scores,
      mode: 'lines',
      name: 'AE Score (MSE)',
      line: { color: THEME.cyan, width: 2 },
    },
  ]

  if (thresholds.some((v) => v > 0)) {
    traces.push({
      x: t,
      y: thresholds,
      mode: 'lines',
      name: 'EMA Threshold (3σ)',
      line: { color: THEME.red, width: 1.5, dash: 'dash' },
    })
  }

  return (
    <Plot
      data={traces}
      layout={{
        paper_bgcolor: THEME.bg,
        plot_bgcolor: THEME.bg,
        font: { color: THEME.text },
        legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
        xaxis: { showgrid: false, title: 'Window' },
        yaxis: { showgrid: true, gridcolor: THEME.border, title: 'MSE' },
        height: 200,
        margin: { l: 40, r: 10, t: 10, b: 40 },
        autosize: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
