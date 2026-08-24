import { useEffect, useMemo, useState } from 'react'

const IN = 'input_tokens_target'
const TTFT = 'ttft_ms_p50'
const TOKS = 'output_tok_s_p50'

function num(row, key, fallback = 0) {
  const v = Number(row?.[key])
  return Number.isFinite(v) ? v : fallback
}

function fmtToks(raw) {
  const v = Number(raw)
  if (!Number.isFinite(v) || v <= 0 || v > 50_000) return '—'
  return v >= 100 ? v.toFixed(0) : v.toFixed(1)
}

function fmtNum(raw, digits = 3) {
  if (raw === null || raw === undefined || raw === '') return '—'
  const v = Number(raw)
  if (!Number.isFinite(v)) return String(raw)
  if (Math.abs(v) >= 100) return v.toFixed(0)
  return Number(v.toPrecision(digits)).toString()
}

function engineFamily(r) {
  const fam = String(r?.engine_family || '').toLowerCase()
  if (fam === 'vllm' || fam === 'sglang') return fam
  const e = String(r?.engine || '').toLowerCase()
  if (e.includes('sglang')) return 'sglang'
  if (e.includes('vllm')) return 'vllm'
  return 'other'
}

function EngineCompareTable({ pairs }) {
  if (!pairs?.length) {
    return (
      <p className="muted">
        No paired vLLM vs SGLang rows yet. Run{' '}
        <code>npm run benchmark:modal:compare:quick</code>.
      </p>
    )
  }

  return (
    <div className="table-wrap compare-wrap">
      <table className="compare-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>GPU</th>
            <th>TP</th>
            <th>In</th>
            <th>Out</th>
            <th>Conc</th>
            <th>Workload</th>
            <th>vLLM TTFT</th>
            <th>SGLang TTFT</th>
            <th>Δ TTFT</th>
            <th>vLLM tok/s</th>
            <th>SGLang tok/s</th>
            <th>Δ tok/s</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((p) => (
            <tr key={`${p.model}-${p.input_tokens}-${p.workload}-${p.concurrency}`}>
              <td className="mono">{p.model}</td>
              <td className="mono">{p.gpu_sku || '—'}</td>
              <td className="mono">{p.tensor_parallel || 1}</td>
              <td className="mono">{p.input_tokens}</td>
              <td className="mono">{p.output_tokens}</td>
              <td className="mono">{p.concurrency}</td>
              <td>{p.workload}</td>
              <td className={`mono ${p.ttft_winner === 'vllm' ? 'win' : ''}`}>
                {num(p.vllm, 'ttft_ms').toFixed(0)} ms
              </td>
              <td className={`mono ${p.ttft_winner === 'sglang' ? 'win' : ''}`}>
                {num(p.sglang, 'ttft_ms').toFixed(0)} ms
              </td>
              <td className="mono">
                {p.ttft_delta_ms > 0 ? '+' : ''}
                {p.ttft_delta_ms?.toFixed(0)} ms
              </td>
              <td className={`mono ${p.tok_s_winner === 'vllm' ? 'win' : ''}`}>
                {fmtToks(p.vllm?.tok_s)}
              </td>
              <td className={`mono ${p.tok_s_winner === 'sglang' ? 'win' : ''}`}>
                {fmtToks(p.sglang?.tok_s)}
              </td>
              <td className="mono">
                {p.tok_s_delta > 0 ? '+' : ''}
                {Number(p.tok_s_delta || 0).toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Bar({ value, max, color = 'var(--teal)' }) {
  const w = max > 0 ? Math.min(100, (100 * value) / max) : 0
  return (
    <div className="track">
      <div className="fill" style={{ '--w': `${w}%`, '--c': color }} />
    </div>
  )
}

function LineChart({ series, title, yLabel, xLabel }) {
  const points = (series || []).flatMap((s) => s.points || [])
  if (points.length < 2) return null
  const W = 640
  const H = 280
  const pad = { l: 52, r: 16, t: 24, b: 42 }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const xmin = Math.min(...xs)
  const xmax = Math.max(...xs) || xmin + 1
  const ymax = Math.max(...ys) * 1.12 || 1
  const sx = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (W - pad.l - pad.r)
  const sy = (y) => pad.t + (1 - y / ymax) * (H - pad.t - pad.b)
  const colors = ['#0d9488', '#0369a1', '#c2410c', '#4d7c0f']

  return (
    <figure className="chart-card">
      <figcaption>
        <strong>{title}</strong>
        <span>
          {yLabel} vs {xLabel}
        </span>
      </figcaption>
      <div className="chart-leg">
        {series.map((s, i) => (
          <span key={s.name} className="leg">
            <i style={{ background: colors[i % colors.length] }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const yv = ymax * t
          const yy = sy(yv)
          return (
            <g key={t}>
              <line x1={pad.l} y1={yy} x2={W - pad.r} y2={yy} className="grid" />
              <text x={pad.l - 8} y={yy + 3} className="tick" textAnchor="end">
                {yv.toFixed(0)}
              </text>
            </g>
          )
        })}
        {series.map((s, i) => {
          const pts = [...(s.points || [])].sort((a, b) => a.x - b.x)
          if (pts.length < 2) return null
          const d = pts
            .map(
              (p, j) =>
                `${j ? 'L' : 'M'} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`,
            )
            .join(' ')
          const color = colors[i % colors.length]
          return (
            <g key={s.name}>
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth="2.6"
                strokeLinecap="round"
                className="line"
              />
              {pts.map((p) => (
                <circle
                  key={`${s.name}-${p.x}`}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r="3.6"
                  fill={color}
                />
              ))}
            </g>
          )
        })}
        {[...new Set(points.map((p) => p.x))].sort((a, b) => a - b).map((xv) => (
          <text
            key={`x-${xv}`}
            x={sx(xv)}
            y={H - pad.b + 18}
            className="tick"
            textAnchor="middle"
          >
            {Number.isInteger(xv) ? xv : xv.toFixed(0)}
          </text>
        ))}
        <text x={W / 2} y={H - 8} className="axis" textAnchor="middle">
          {xLabel}
        </text>
      </svg>
    </figure>
  )
}

function ConcurrencyPanel({ panel }) {
  if (!panel) return null
  const title = `${panel.model} · S=${panel.input_tokens} · ${panel.gpu_sku} · TP=${panel.tensor_parallel}`
  return (
    <article className="concurrency-panel" id={`conc-${panel.id}`}>
      <header>
        <h3>{title}</h3>
        <p>{panel.fabric_note} · workload={panel.workload}</p>
      </header>
      <div className="concurrency-charts">
        <LineChart
          series={panel.ttft_series || []}
          title="Latency vs concurrency"
          yLabel="TTFT (ms)"
          xLabel="Concurrency"
        />
        <LineChart
          series={panel.toks_series || []}
          title="Throughput vs concurrency"
          yLabel="tok/s"
          xLabel="Concurrency"
        />
      </div>
    </article>
  )
}

function ConcurrencyPanels({ panels }) {
  if (!panels?.length) return null
  return (
    <section id="concurrency" className="concurrency-section">
      <h2>Capacity · concurrency sweeps</h2>
      <p className="lead">
        Fixed input shape (single workload) — how TTFT and aggregate tok/s scale as
        batch concurrency rises. TP=2 panels use A100×2 NCCL; not an NVLink on/off A/B.
      </p>
      {panels.map((panel) => (
        <ConcurrencyPanel key={panel.id} panel={panel} />
      ))}
    </section>
  )
}

function CacheChart({ rows }) {
  if (!rows?.length) return null
  const W = 640
  const H = 250
  const pad = { l: 52, r: 16, t: 16, b: 44 }
  const max = Math.max(...rows.flatMap((r) => [r.cold, r.warm])) * 1.15 || 1
  const group = (W - pad.l - pad.r) / rows.length
  const barW = group * 0.32

  return (
    <figure className="chart-card">
      <figcaption>
        <strong>Prefix cache on Modal vLLM</strong>
        <span>Cold vs warm TTFT (ms)</span>
      </figcaption>
      <div className="chart-leg">
        <span className="leg">
          <i style={{ background: '#0f766e' }} />
          cold_prefix
        </span>
        <span className="leg">
          <i style={{ background: '#d97706' }} />
          warm_prefix
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Prefix cache">
        {rows.map((r, i) => {
          const x0 = pad.l + i * group + group * 0.18
          const coldH = (r.cold / max) * (H - pad.t - pad.b)
          const warmH = (r.warm / max) * (H - pad.t - pad.b)
          return (
            <g key={r.tokens}>
              <rect
                className="bar-grow"
                x={x0}
                y={H - pad.b - coldH}
                width={barW}
                height={coldH}
                fill="#0f766e"
                rx="3"
                style={{ animationDelay: `${i * 0.07}s` }}
              />
              <rect
                className="bar-grow"
                x={x0 + barW + 6}
                y={H - pad.b - warmH}
                width={barW}
                height={warmH}
                fill="#d97706"
                rx="3"
                style={{ animationDelay: `${i * 0.07 + 0.04}s` }}
              />
              <text
                x={x0 + barW + 3}
                y={H - 14}
                className="tick"
                textAnchor="middle"
              >
                {r.tokens}
              </text>
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

function DiscoveryPanel({ discovery }) {
  const [active, setActive] = useState(discovery?.scenarios?.[0]?.id ?? '')
  const scenario =
    discovery?.scenarios?.find((s) => s.id === active) ??
    discovery?.scenarios?.[0]

  if (!discovery?.scenarios?.length) return null

  return (
    <section id="discover" className="discover">
      <div className="discover-intro">
        <p className="eyebrow">Start here</p>
        <h2>{discovery.purpose?.headline}</h2>
        <p className="lead">{discovery.purpose?.subhead}</p>
        {discovery.purpose?.audience && (
          <p className="audience">{discovery.purpose.audience}</p>
        )}
      </div>

      <div className="discover-questions">
        <h3>Five questions before you size a deal</h3>
        <ol>
          {discovery.questions.map((q) => (
            <li key={q.id}>
              <strong>{q.question}</strong>
              <span>{q.why}</span>
            </li>
          ))}
        </ol>
      </div>

      <div className="discover-scenarios">
        <h3>Pick a workload — see what you&apos;d discover</h3>
        <div className="scenario-tabs" role="tablist" aria-label="Workload scenarios">
          {discovery.scenarios.map((s) => (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={active === s.id}
              className={active === s.id ? 'active' : ''}
              onClick={() => setActive(s.id)}
            >
              {s.title}
            </button>
          ))}
        </div>

        {scenario && (
          <article className="scenario-card rise">
            <header>
              <h4>{scenario.title}</h4>
              <p>{scenario.subtitle}</p>
            </header>
            <div className="scenario-profile">
              <span>S={scenario.profile?.input_tokens?.toLocaleString()}</span>
              <span>out={scenario.profile?.output_tokens}</span>
              <span>conc={scenario.profile?.concurrency}</span>
              <span>cache={scenario.profile?.cache}</span>
            </div>
            <div className="scenario-findings">
              {scenario.discoveries.map((d) => (
                <div className="finding" key={d.label}>
                  <span className="fk">{d.label}</span>
                  <span className="fv">{d.value}</span>
                  <span className="fd">{d.detail}</span>
                  {d.anchor && (
                    <a className="fa" href={d.anchor}>
                      View →
                    </a>
                  )}
                </div>
              ))}
            </div>
            <p className="scenario-rec">
              <strong>Recommendation:</strong> {scenario.recommendation}
            </p>
          </article>
        )}
      </div>
    </section>
  )
}

export default function Report() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    let cancelled = false
    fetch('/benchmark-data.json')
      .then((r) => {
        if (!r.ok) throw new Error(`benchmark-data.json ${r.status}`)
        return r.json()
      })
      .then((json) => {
        if (!cancelled) setData(json)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const probes = useMemo(
    () =>
      (data?.hardware || []).filter(
        (r) => r.workload === 'hardware_probe',
      ),
    [data],
  )
  const ncclRows = useMemo(
    () =>
      (data?.hardware || []).filter(
        (r) => r.workload === 'nccl_allreduce',
      ),
    [data],
  )
  const inference = useMemo(
    () => (data?.rows || []).filter((r) => r.stack_layer !== 'hardware'),
    [data],
  )
  const filtered = useMemo(() => {
    return inference.filter((r) => {
      if (filter === 'all') return true
      if (filter === 'live') return true
      if (filter.startsWith('layer:')) return r.stack_layer === filter.slice(6)
      if (filter.startsWith('provider:')) return r.provider === filter.slice(9)
      if (filter.startsWith('engine:')) return engineFamily(r) === filter.slice(7)
      if (filter.startsWith('gpu:')) return (r.gpu_sku || '') === filter.slice(4)
      return true
    })
  }, [inference, filter])

  const modelLive = useMemo(
    () => (data?.rows || []).filter((r) => r.stack_layer === 'model'),
    [data],
  )
  const servingLive = useMemo(
    () =>
      (data?.rows || []).filter((r) => r.stack_layer === 'serving'),
    [data],
  )

  const maxMemcpy = Math.max(1, ...probes.map((r) => num(r, 'memcpy_gbps')))
  const maxTflops = Math.max(
    1,
    ...probes.map((r) => num(r, 'matmul_tflops_bf16')),
  )
  const maxModelTtft = Math.max(1, ...modelLive.map((r) => num(r, TTFT)))
  const maxServeTtft = Math.max(1, ...servingLive.map((r) => num(r, TTFT)))

  if (error) {
    return (
      <div className="report shell">
        <p className="error">
          Could not load results. Run <code>npm run report</code> then refresh.
          <br />
          {error}
        </p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="report shell">
        <div className="loading">
          <div className="pulse" />
          Loading benchmark matrix…
        </div>
      </div>
    )
  }

  const { summary, insights, coverage, inventory, model_series, cache_compare, concurrency_panels, discovery, engine_compare, filters } =
    data
  const providers = summary.providers || []
  const engines = filters?.engines || summary.engines || []
  const gpuSkus = filters?.gpu_skus || summary.gpu_skus || []

  return (
    <div className="report">
      <div className="aurora" aria-hidden="true" />
      <nav className="topnav">
        <div className="topnav-inner">
          <a className="brand" href="/">
            KV <span>Cache</span>
          </a>
          <div className="nav-links">
            <a href="#discover">Discover</a>
            <a href="#coverage">Coverage</a>
            <a href="#insights">Insights</a>
            <a href="#hardware">Hardware</a>
            <a href="#model">Model</a>
            <a href="#serving">Serving</a>
            <a href="#concurrency">Concurrency</a>
            <a href="#compare">Engine A/B</a>
            <a href="#matrix">Matrix</a>
            <a href="/">Explorer</a>
          </div>
        </div>
      </nav>

      <main className="shell">
        <header className="hero">
          <p className="eyebrow">Inference customer discovery</p>
          <h1 className="hero-brand">
            KV <em>Cache</em>
          </h1>
          <p className="hero-sub">
            Profile workload shape on Modal — CUDA / NCCL hardware, model prefill,
            vLLM vs SGLang serving with concurrency, prefix cache, and TP=2 (A100×2).
          </p>
          <div className="hero-cta">
            <a className="btn primary" href="#discover">
              Start discovery
            </a>
            <a className="btn ghost" href="#matrix">
              Full matrix
            </a>
          </div>

          <div className="stack-strip">
            <article className="stack-card rise" style={{ '--d': '0s' }}>
              <div className="step">01 · Hardware</div>
              <h3>CUDA + NCCL</h3>
              <p>Memcpy, GEMM TFLOPS, multi-GPU all-reduce on Modal A100×2.</p>
            </article>
            <article className="stack-card rise" style={{ '--d': '0.08s' }}>
              <div className="step">02 · Model</div>
              <h3>KV &amp; prefill</h3>
              <p>Batch=1 transformers — GQA GiB and TTFT ≈ prefill.</p>
            </article>
            <article className="stack-card rise" style={{ '--d': '0.16s' }}>
              <div className="step">03 · Serving</div>
              <h3>vLLM vs SGLang</h3>
              <p>Same matrix on Modal A100 — engine A/B, prefix cache, TP=2 NCCL.</p>
            </article>
          </div>

          <div className="kpi rise" style={{ '--d': '0.22s' }}>
            <div className="stat">
              <span className="k">Total</span>
              <span className="v">{summary.total}</span>
            </div>
            <div className="stat">
              <span className="k">Hardware</span>
              <span className="v">{summary.hardware}</span>
            </div>
            <div className="stat">
              <span className="k">Model</span>
              <span className="v">{summary.model}</span>
            </div>
            <div className="stat">
              <span className="k">Serving</span>
              <span className="v">{summary.serving}</span>
            </div>
            <div className="stat">
              <span className="k">GQA</span>
              <span className="v">7×</span>
            </div>
          </div>
        </header>

        <DiscoveryPanel discovery={discovery} />

        <section id="coverage">
          <h2>Coverage</h2>
          <p className="lead">What this report covers.</p>
          <div className="cov-grid">
            {coverage.map((c) => (
              <article key={c.id} className="cov ok">
                <div className="cov-top">
                  <strong>{c.title}</strong>
                </div>
                <p>{c.detail}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="insights">
          <h2>Headline insights</h2>
          <p className="lead">The story without opening the CSV.</p>
          <div className="kpi">
            {insights.map((ins) => (
              <div className="stat insight" key={ins.label}>
                <span className="k">{ins.label}</span>
                <span className="v">{ins.value}</span>
                <span className="s">{ins.sub}</span>
              </div>
            ))}
          </div>
        </section>

        <section id="hardware">
          <h2>Hardware layer</h2>
          <p className="lead">
            CUDA microbench on Modal (device memcpy + GEMM) and multi-GPU{' '}
            <strong>NCCL all-reduce</strong> — the floor under decode and TP.
          </p>
          <div className="hw-grid">
            {[...probes]
              .sort((a, b) => num(b, 'memcpy_gbps') - num(a, 'memcpy_gbps'))
              .map((r) => (
                <article className="hw-card" key={r.run_id || r.gpu_sku}>
                  <div className="sku">{r.gpu_sku || '?'}</div>
                  <div className="name">
                    {r.gpu_name || r.model_short} · {fmtNum(r.memory_gib)} GiB ·
                    CC {r.compute_capability || '—'}
                  </div>
                  <div className="hw-metrics">
                    <div>
                      <span>Memcpy</span>
                      <b>{fmtNum(r.memcpy_gbps)} GB/s</b>
                    </div>
                    <div>
                      <span>BF16</span>
                      <b>{fmtNum(r.matmul_tflops_bf16)} TFLOPS</b>
                    </div>
                    <div>
                      <span>FP16</span>
                      <b>{fmtNum(r.matmul_tflops_fp16)} TFLOPS</b>
                    </div>
                    <div>
                      <span>SMs</span>
                      <b>{r.sm_count || '—'}</b>
                    </div>
                  </div>
                </article>
              ))}
            {ncclRows.map((r) => (
              <article className="hw-card nccl" key={r.run_id || 'nccl'}>
                <div className="sku">{r.gpu_sku || 'NCCL'}</div>
                <div className="name">
                  NCCL all-reduce · world {r.nccl_world_size || 2}
                </div>
                <div className="hw-metrics">
                  <div>
                    <span>Bus BW</span>
                    <b>{fmtNum(r.nccl_busbw_gbps)} GB/s</b>
                  </div>
                  <div>
                    <span>Alg BW</span>
                    <b>{fmtNum(r.nccl_algbw_gbps)} GB/s</b>
                  </div>
                </div>
              </article>
            ))}
          </div>

          {probes.length > 0 && (
            <>
              <h3>Memcpy bandwidth</h3>
              {[...probes]
                .sort((a, b) => num(a, 'memcpy_gbps') - num(b, 'memcpy_gbps'))
                .map((r) => (
                  <div className="bar-row" key={`m-${r.gpu_sku}`}>
                    <span className="lbl">{r.gpu_sku}</span>
                    <Bar
                      value={num(r, 'memcpy_gbps')}
                      max={maxMemcpy}
                      color="var(--sky)"
                    />
                    <span className="val">
                      {num(r, 'memcpy_gbps').toFixed(0)} GB/s
                    </span>
                  </div>
                ))}
              <h3>BF16 GEMM</h3>
              {[...probes]
                .sort(
                  (a, b) =>
                    num(a, 'matmul_tflops_bf16') -
                    num(b, 'matmul_tflops_bf16'),
                )
                .map((r) => (
                  <div className="bar-row" key={`t-${r.gpu_sku}`}>
                    <span className="lbl">{r.gpu_sku}</span>
                    <Bar
                      value={num(r, 'matmul_tflops_bf16')}
                      max={maxTflops}
                      color="var(--teal)"
                    />
                    <span className="val">
                      {num(r, 'matmul_tflops_bf16').toFixed(1)} TFLOPS
                    </span>
                  </div>
                ))}
            </>
          )}
        </section>

        <section id="inventory">
          <h2>Inventory</h2>
          {['providers', 'engines', 'gpu_skus', 'models', 'workloads'].map((key) => (
            <div key={key}>
              <h3>{key === 'gpu_skus' ? 'GPU SKUs' : key[0].toUpperCase() + key.slice(1)}</h3>
              <div className="pills">
                {Object.entries(inventory[key] || {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <span className="pill" key={k}>
                      {k} · {v}
                    </span>
                  ))}
              </div>
            </div>
          ))}
        </section>

        <section id="model">
          <h2>Model layer</h2>
          <p className="lead">
            Architecture &amp; KV on Modal — batch=1, no HTTP. TTFT ≈ prefill.
          </p>
          <div className="charts">
            <LineChart
              series={model_series || []}
              title="Model-layer TTFT vs context"
              yLabel="TTFT (ms)"
              xLabel="Input tokens"
            />
          </div>
          {[...modelLive]
            .sort(
              (a, b) =>
                num(a, IN) - num(b, IN) ||
                String(a.model_short).localeCompare(String(b.model_short)),
            )
            .map((r) => (
              <div
                className="bar-row"
                key={r.run_id || `${r.model_short}-${r[IN]}`}
              >
                <span className="lbl">
                  {String(r.model_short).slice(0, 16)} · S={r[IN]}
                </span>
                <Bar
                  value={num(r, TTFT)}
                  max={maxModelTtft}
                  color="var(--sky)"
                />
                <span className="val">
                  {num(r, TTFT).toFixed(0)} ms · {fmtToks(r[TOKS])} tok/s · KV{' '}
                  {fmtNum(r.kv_gib_modeled_gqa)}
                </span>
              </div>
            ))}
        </section>

        <section id="serving">
          <h2>Serving layer</h2>
          <p className="lead">
            vLLM and SGLang on Modal — A100 (TP=1, 7B) and A100×2 (TP=2, 32B · NCCL).
            Prefix cache and concurrency sweeps on identical workload matrices.
          </p>
          <div className="charts">
            <CacheChart rows={cache_compare || []} />
          </div>
          <ConcurrencyPanels panels={concurrency_panels} />
          {[...servingLive]
            .sort(
              (a, b) =>
                engineFamily(a).localeCompare(engineFamily(b)) ||
                num(a, IN) - num(b, IN) ||
                num(a, 'concurrency') - num(b, 'concurrency'),
            )
            .map((r) => (
              <div
                className="bar-row"
                key={
                  r.run_id ||
                  `${r.provider}-${engineFamily(r)}-${r[IN]}-${r.workload}-${r.concurrency}`
                }
              >
                <span className="lbl">
                  {engineFamily(r)} · {String(r.gpu_sku || 'A100').slice(0, 8)} ·{' '}
                  {String(r.model_short).slice(0, 10)} · S={r[IN]} · c=
                  {r.concurrency || 1} · {r.workload || 'single'}
                </span>
                <Bar
                  value={num(r, TTFT)}
                  max={maxServeTtft}
                  color={engineFamily(r) === 'sglang' ? 'var(--amber)' : 'var(--teal)'}
                />
                <span className="val">
                  {num(r, TTFT).toFixed(0)} ms · {fmtToks(r[TOKS])} tok/s · cache{' '}
                  {r.cached_prompt_tokens || 0}
                </span>
              </div>
            ))}
        </section>

        <section id="compare">
          <h2>Engine A/B · vLLM vs SGLang</h2>
          <p className="lead">
            Paired rows — same model, input/output shape, concurrency, workload, GPU, and TP.
            Lower TTFT and higher tok/s win (highlighted).
          </p>
          <EngineCompareTable pairs={engine_compare || []} />
        </section>

        <section id="matrix">
          <h2>Results matrix</h2>
          <p className="lead">
            {filtered.length} rows · filter by stack, engine, GPU, or provider
          </p>
          <div className="filters" role="group" aria-label="Filter">
            {[
              ['all', 'All'],
              ['layer:model', 'Model'],
              ['layer:serving', 'Serving'],
  
              ...engines.map((e) => [`engine:${e}`, e === 'vllm' ? 'vLLM' : 'SGLang']),
              ...gpuSkus.map((g) => [`gpu:${g}`, g]),
              ...providers.map((p) => [`provider:${p}`, p]),
            ].map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={filter === id ? 'active' : ''}
                onClick={() => setFilter(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Stack</th>
                  <th>Provider</th>
                  <th>Model</th>
                  <th>Engine</th>
                  <th>GPU</th>
                  <th>In</th>
                  <th>Out</th>
                  <th>Conc</th>
                  <th>TP</th>
                  <th>Workload</th>
                  <th>TTFT</th>
                  <th>Tok/s</th>
                  <th>Cached</th>
                  <th>KV GQA</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.run_id || JSON.stringify(r).slice(0, 48)}>
                    <td>{r.stack_layer}</td>
                    <td>{r.provider}</td>
                    <td className="mono">{r.model_short}</td>
                    <td className="mono">
                      {engineFamily(r) === 'sglang'
                        ? 'SGLang'
                        : engineFamily(r) === 'vllm'
                          ? 'vLLM'
                          : String(r.engine || '').slice(0, 24)}
                    </td>
                    <td className="mono">{r.gpu_sku || '—'}</td>
                    <td className="mono">{r[IN]}</td>
                    <td className="mono">{r.output_tokens_target}</td>
                    <td className="mono">{r.concurrency || 1}</td>
                    <td className="mono">{r.tensor_parallel || 1}</td>
                    <td>{r.workload}</td>
                    <td className="mono">{num(r, TTFT).toFixed(0)}</td>
                    <td className="mono">{fmtToks(r[TOKS])}</td>
                    <td className="mono">{r.cached_prompt_tokens || 0}</td>
                    <td className="mono">{fmtNum(r.kv_gib_modeled_gqa)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="reproduce">
          <h2>Reproduce</h2>
          <pre className="code">
            <span className="c"># Full stack on Modal (CUDA + NCCL + TP=2 vLLM)</span>
            {'\n'}npm run benchmark:full
            {'\n\n'}
            <span className="c"># Or step by step</span>
            {'\n'}npm run benchmark:modal:hardware:nccl
            {'\n'}npm run benchmark:modal
            {'\n'}npm run benchmark:modal:vllm
            {'\n'}npm run benchmark:modal:sglang
            {'\n'}npm run benchmark:modal:compare:quick
            {'\n'}npm run benchmark:modal:vllm:tp2
            {'\n\n'}
            <span className="c"># Refresh this React report</span>
            {'\n'}npm run report && npm run build
          </pre>
        </section>

        <footer>
          <span>Generated {data.generated_at}</span>
          <span>Hardware → model → serving · Modal only</span>
          <a href="/">← KV explorer</a>
        </footer>
      </main>
    </div>
  )
}
