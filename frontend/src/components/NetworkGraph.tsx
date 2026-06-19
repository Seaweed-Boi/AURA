import Plot from 'react-plotly.js'
import type { NodeInfo } from '../types'
import { THEME } from '../theme'

interface Props {
  nodes: NodeInfo[]
  edgeIndex: number[][] | null
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Trace = any

export default function NetworkGraph({ nodes, edgeIndex }: Props) {
  const n = nodes.length || 20
  const angles = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n)
  const posX = angles.map((a) => Math.cos(a))
  const posY = angles.map((a) => Math.sin(a))

  const edgeTraces: Trace[] = []
  if (edgeIndex && edgeIndex.length === 2) {
    const [src, dst] = edgeIndex
    for (let idx = 0; idx < Math.min(src.length, 60); idx++) {
      const s = src[idx]
      const d = dst[idx]
      edgeTraces.push({
        x: [posX[s], posX[d], null],
        y: [posY[s], posY[d], null],
        mode: 'lines',
        line: { width: 1, color: '#1e3a5a' },
        hoverinfo: 'none',
        showlegend: false,
      })
    }
  }

  const nodeTrace: Trace = {
    x: posX,
    y: posY,
    mode: 'markers+text',
    marker: {
      size: nodes.map((nd) => (nd.critical ? 20 : 14)),
      color: nodes.map((nd) => nd.color),
      symbol: nodes.map((nd) => (nd.critical ? 'diamond' : 'circle')),
      line: { width: 1.5, color: '#ffffff' },
    },
    text: nodes.map((nd) => `N${nd.index}`),
    textposition: 'top center',
    textfont: { size: 8, color: THEME.text },
    hovertext: nodes.map((nd) => `${nd.label}<br>(${nd.state})`),
    hoverinfo: 'text',
    showlegend: false,
  }

  return (
    <Plot
      data={[...edgeTraces, nodeTrace]}
      layout={{
        paper_bgcolor: THEME.bg,
        plot_bgcolor: THEME.bg,
        margin: { l: 10, r: 10, t: 10, b: 10 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
        height: 320,
        autosize: true,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
