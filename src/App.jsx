import { useMemo, useState } from 'react'
import { MODELS, GPUS, PRECISION } from './config'
import { KvSequenceChart } from './KvSequenceChart'
import {
  bytesToGiB,
  computeDenseKvCacheBytes,
  computeGqaKvCacheBytes,
  formatBytes,
  headDimFromModel,
  kvReductionRatio,
} from './kvCache'
import './App.css'

const BATCH_MIN = 1
const BATCH_MAX = 256
const TOK_MIN = 100
const TOK_MAX = 200_000

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}

export default function App() {
  const [modelId, setModelId] = useState('qwen')
  const [gpuId, setGpuId] = useState('doH100')
  const [batch, setBatch] = useState(4)
  const [inputTokens, setInputTokens] = useState(2048)
  const [outputTokens, setOutputTokens] = useState(512)
  const [precisionId, setPrecisionId] = useState('bf16')

  const model = MODELS[modelId]
  const gpu = GPUS[gpuId]
  const precision = PRECISION[precisionId]
  const headDim = headDimFromModel(model)
  const kvRatio = kvReductionRatio(model)

  const sequenceLength = inputTokens + outputTokens

  const denseKvBytes = useMemo(
    () =>
      computeDenseKvCacheBytes({
        batch,
        sequenceLength,
        layers: model.layers,
        hiddenDim: model.hiddenDim,
        bytesPerElement: precision.bytesPerElement,
      }),
    [
      batch,
      sequenceLength,
      model.layers,
      model.hiddenDim,
      precision.bytesPerElement,
    ],
  )

  const gqaKvBytes = useMemo(
    () =>
      computeGqaKvCacheBytes({
        batch,
        sequenceLength,
        layers: model.layers,
        numKvHeads: model.numKvHeads,
        headDim,
        bytesPerElement: precision.bytesPerElement,
      }),
    [
      batch,
      sequenceLength,
      model.layers,
      model.numKvHeads,
      headDim,
      precision.bytesPerElement,
    ],
  )

  const gpuBytes = gpu.memoryGb * 1024 ** 3
  const gqaGiB = bytesToGiB(gqaKvBytes)
  const pctGqa = (gqaKvBytes / gpuBytes) * 100
  const overBudget = gqaKvBytes > gpuBytes

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-top">
          <div>
            <p className="eyebrow">LLM inference · memory & benchmarks</p>
            <h1>KV cache explorer</h1>
            <p className="lede">
              Model KV footprint (dense vs GQA), VRAM budget, and a benchmark
              harness for GPU droplets and APIs — built for production inference
              planning on cloud GPUs.
            </p>
          </div>
          <nav className="hero-nav">
            <a className="nav-chip" href="/benchmark-report.html">
              Benchmark report →
            </a>
          </nav>
        </div>
      </header>

      <div className="layout">
        <section className="panel controls">
          <h2>Inputs</h2>

          <label className="field">
            <span>Model (OSS presets)</span>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
            >
              {Object.values(MODELS).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} — {m.blurb}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>GPU / accelerator</span>
            <select value={gpuId} onChange={(e) => setGpuId(e.target.value)}>
              {Object.values(GPUS).map((g) => (
                <option key={g.id} value={g.id}>
                  {g.name} ({g.detail})
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            <span>KV dtype</span>
            <select
              value={precisionId}
              onChange={(e) => setPrecisionId(e.target.value)}
            >
              {Object.values(PRECISION).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label} ({p.bytesPerElement} byte
                  {p.bytesPerElement > 1 ? 's' : ''} / element)
                </option>
              ))}
            </select>
          </label>

          <div className="field">
            <div className="field-head">
              <span>Batch size</span>
              <span className="mono val">{batch}</span>
            </div>
            <input
              type="range"
              min={BATCH_MIN}
              max={BATCH_MAX}
              value={batch}
              onChange={(e) => setBatch(Number(e.target.value))}
            />
            <div className="ticks-labels">
              <span>{BATCH_MIN}</span>
              <span>{BATCH_MAX}</span>
            </div>
          </div>

          <div className="grid2">
            <label className="field">
              <span>Input tokens</span>
              <input
                type="number"
                min={TOK_MIN}
                max={TOK_MAX}
                value={inputTokens}
                onChange={(e) =>
                  setInputTokens(
                    clamp(Number(e.target.value) || 0, TOK_MIN, TOK_MAX),
                  )
                }
              />
            </label>
            <label className="field">
              <span>Output tokens (max new)</span>
              <input
                type="number"
                min={TOK_MIN}
                max={TOK_MAX}
                value={outputTokens}
                onChange={(e) =>
                  setOutputTokens(
                    clamp(Number(e.target.value) || 0, TOK_MIN, TOK_MAX),
                  )
                }
              />
            </label>
          </div>

          <p className="hint">
            <span className="mono">S = {sequenceLength}</span> tokens ·{' '}
            {model.numAttentionHeads} Q heads · {model.numKvHeads} KV heads ·
            head dim {headDim}
          </p>
        </section>

        <section className="panel viz">
          <h2>KV cache size</h2>

          <div className="stat-duo">
            <div className="stat-block primary-stat">
              <p className="stat-label">GQA (deployed shape)</p>
              <p className="stat-main">{formatBytes(gqaKvBytes)}</p>
              <p className="stat-sub mono">
                {gqaGiB.toFixed(3)} GiB · {pctGqa.toFixed(1)}% of {gpu.name}{' '}
                {overBudget && '· over budget'}
              </p>
            </div>
            <div className="stat-block">
              <p className="stat-label">Dense upper bound</p>
              <p className="stat-main muted">{formatBytes(denseKvBytes)}</p>
              <p className="stat-sub mono">
                {kvRatio.toFixed(0)}× larger than GQA for this architecture
              </p>
            </div>
          </div>

          <KvSequenceChart
            batch={batch}
            layers={model.layers}
            hiddenDim={model.hiddenDim}
            numKvHeads={model.numKvHeads}
            headDim={headDim}
            bytesPerElement={precision.bytesPerElement}
            gpuBytes={gpuBytes}
            gpuName={gpu.name}
            gpuMemoryGb={gpu.memoryGb}
            currentSequenceLength={sequenceLength}
            currentDenseKvBytes={denseKvBytes}
            currentGqaKvBytes={gqaKvBytes}
            inputTokens={inputTokens}
            outputTokens={outputTokens}
            onInputTokensChange={(n) =>
              setInputTokens(clamp(n, TOK_MIN, TOK_MAX))
            }
            onOutputTokensChange={(n) =>
              setOutputTokens(clamp(n, TOK_MIN, TOK_MAX))
            }
          />

          <div className="formula-card">
            <p className="formula-title">Formulas (per layer, per token)</p>
            <code className="formula">
              GQA: {batch} × {sequenceLength} × {model.layers} × 2 ×{' '}
              {model.numKvHeads} × {headDim} × {precision.bytesPerElement} ={' '}
              {formatBytes(gqaKvBytes)}
            </code>
            <code className="formula">
              Dense: {batch} × {sequenceLength} × {model.layers} × 2 ×{' '}
              {model.hiddenDim} × {precision.bytesPerElement} ={' '}
              {formatBytes(denseKvBytes)}
            </code>
          </div>
        </section>
      </div>

      <section className="explain-panel" aria-labelledby="explain-kv-heading">
        <h2 id="explain-kv-heading">Why two numbers?</h2>
        <p>
          Real stacks like <strong>{model.name}</strong> use{' '}
          <strong>grouped-query attention (GQA)</strong>: many query heads share
          fewer KV heads, so the cache stores{' '}
          <span className="mono">2 × n_kv × d</span> per position, not{' '}
          <span className="mono">2 × hidden</span>. LMCache, vLLM, and provider
          calculators use the <strong>GQA</strong> line; textbook dense formulas
          are a <strong>{kvRatio.toFixed(0)}× planning ceiling</strong> for this
          model.
        </p>
        <p>
          The bundled <a href="/benchmark-report.html">benchmark report</a>{' '}
          splits results into two tiers: <strong>model layer</strong>{' '}
          (isolated GPU — KV sizing, prefill TTFT, decode tok/s via Modal) and{' '}
          <strong>serving layer</strong> (HTTP API, vLLM under load, prefix
          cache hits). See <code>docs/LAYERS.md</code> and{' '}
          <code>docs/DEPLOY_DIGITALOCEAN.md</code> for how to run each on a{' '}
          <strong>DigitalOcean GPU droplet</strong> or provider API.
        </p>
        <p>
          For larger models, use <code>benchmarks/scripts/serve_vllm.sh</code>{' '}
          with tensor parallel (NCCL) or AWQ/FP8 quantization — see{' '}
          <code>docs/GPU_STACK.md</code>.
        </p>
      </section>

      <footer className="foot">
        Planning tool + benchmark harness. GQA/MQA/paged KV may reduce further;
        activations and weights are not included.
      </footer>
    </div>
  )
}
