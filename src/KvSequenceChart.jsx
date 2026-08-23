import { useEffect, useRef, useState } from 'react'
import { computeDenseKvCacheBytes, computeGqaKvCacheBytes } from './kvCache'

const S_AXIS_MIN = 100
const S_AXIS_MAX = 200_000
const LOG_S_MIN = Math.log10(S_AXIS_MIN)
const LOG_S_MAX = Math.log10(S_AXIS_MAX)
/** X-axis tick positions (tokens), log-friendly multiplicative steps. */
const X_AXIS_TICK_TOKENS = [100, 1000, 10000, 50000, 100000, 200000]
const MIB = 1024 ** 2
const GIB = 1024 ** 3
/** Fixed Y-axis: 100 MiB–250 GiB (binary), log₁₀ scale on screen. */
const Y_AXIS_MIN_BYTES = 100 * MIB
const Y_AXIS_MAX_BYTES = 250 * GIB
const LOG_Y_MIN = Math.log10(Y_AXIS_MIN_BYTES)
const LOG_Y_MAX = Math.log10(Y_AXIS_MAX_BYTES)
/** Tick positions (bytes), ~even spacing in log space; labels via formatYAxisTick. */
const Y_AXIS_TICK_BYTES = [
  100 * MIB,
  250 * MIB,
  500 * MIB,
  1 * GIB,
  2 * GIB,
  4 * GIB,
  8 * GIB,
  16 * GIB,
  32 * GIB,
  64 * GIB,
  128 * GIB,
  250 * GIB,
]
/** Segments for KV curve; sample S log-uniformly (KV ∝ S → straight in log–log). */
const KV_CURVE_STEPS = 160

/** Match App.jsx token clamps for input/output when scrubbing S. */
const TOK_MIN = 100
const TOK_MAX = 200_000
const S_SUM_MIN = TOK_MIN * 2
const S_SUM_MAX = TOK_MAX * 2

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}

/**
 * Map target total S = input + output onto the pair: shrink output to TOK_MIN first,
 * then input; grow output until TOK_MAX, then input.
 */
function allocateTokensForTotalS(inCur, outCur, sTargetRaw) {
  const sHi = Math.min(S_AXIS_MAX, S_SUM_MAX)
  const s = clamp(Math.round(sTargetRaw), S_SUM_MIN, sHi)
  const sCur = inCur + outCur
  if (s === sCur) {
    return { in: inCur, out: outCur }
  }
  if (s < sCur) {
    const dec = sCur - s
    const takeOut = Math.min(outCur - TOK_MIN, dec)
    const outNew = outCur - takeOut
    const decRem = dec - takeOut
    const takeIn = Math.min(inCur - TOK_MIN, decRem)
    const inNew = inCur - takeIn
    return { in: inNew, out: outNew }
  }
  const inc = s - sCur
  const addOut = Math.min(TOK_MAX - outCur, inc)
  const outNew = outCur + addOut
  const incRem = inc - addOut
  const addIn = Math.min(TOK_MAX - inCur, incRem)
  const inNew = inCur + addIn
  return { in: inNew, out: outNew }
}

function formatXAxisTick(tokens) {
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}k`
  return String(tokens)
}

/** Y-axis tick label with MiB / GiB suffix (binary units). */
function formatYAxisTick(bytes) {
  if (bytes < GIB) {
    return `${Math.round(bytes / MIB)} MiB`
  }
  return `${Math.round(bytes / GIB)} GiB`
}

function kvDenseAtS(batch, S, layers, hiddenDim, bytesPerElement) {
  return computeDenseKvCacheBytes({
    batch,
    sequenceLength: Math.round(S),
    layers,
    hiddenDim,
    bytesPerElement,
  })
}

function kvGqaAtS(batch, S, layers, numKvHeads, headDim, bytesPerElement) {
  return computeGqaKvCacheBytes({
    batch,
    sequenceLength: Math.round(S),
    layers,
    numKvHeads,
    headDim,
    bytesPerElement,
  })
}

export function KvSequenceChart({
  batch,
  layers,
  hiddenDim,
  numKvHeads,
  headDim,
  bytesPerElement,
  gpuBytes,
  gpuName,
  gpuMemoryGb,
  currentSequenceLength,
  currentDenseKvBytes,
  currentGqaKvBytes,
  inputTokens,
  outputTokens,
  onInputTokensChange,
  onOutputTokensChange,
}) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  /** Plot geometry + log-S mapping (device pixels), updated each paint for hit-testing. */
  const layoutRef = useRef(null)
  const inputTokensRef = useRef(inputTokens)
  const outputTokensRef = useRef(outputTokens)
  const onInputRef = useRef(onInputTokensChange)
  const onOutputRef = useRef(onOutputTokensChange)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    inputTokensRef.current = inputTokens
  }, [inputTokens])

  useEffect(() => {
    outputTokensRef.current = outputTokens
  }, [outputTokens])

  useEffect(() => {
    onInputRef.current = onInputTokensChange
  }, [onInputTokensChange])

  useEffect(() => {
    onOutputRef.current = onOutputTokensChange
  }, [onOutputTokensChange])

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const padR = 16
    const padT = 16
    const padB = 40

    let resizeRaf = 0

    const paint = () => {
      /* Size from layout only — do not set canvas inline width/height (avoids grid min-content
       * fighting the bitmap width and retriggering ResizeObserver in a loop). */
      const wCss = Math.max(1, Math.floor(canvas.clientWidth))
      const hCss = 280
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const W = Math.floor(wCss * dpr)
      const H = Math.floor(hCss * dpr)
      canvas.width = W
      canvas.height = H

      /* Left: rotated title outside plot + y tick numbers. */
      const padYTitle = Math.round(22 * dpr)
      const padYNums = Math.round(58 * dpr)
      const padL = padYTitle + padYNums

      const plotW = W - padL - padR
      const plotH = H - padT - padB
      const ox = padL
      const oy = padT

      const minY = Y_AXIS_MIN_BYTES
      const maxY = Y_AXIS_MAX_BYTES

      const xToPx = (s) => {
        const clamped = Math.min(S_AXIS_MAX, Math.max(S_AXIS_MIN, s))
        const t =
          (Math.log10(clamped) - LOG_S_MIN) / Math.max(LOG_S_MAX - LOG_S_MIN, 1e-9)
        return ox + t * plotW
      }
      const yToPx = (kv) => {
        const clamped = Math.min(Math.max(kv, minY), maxY)
        const lv = Math.log10(clamped)
        const t = (lv - LOG_Y_MIN) / Math.max(LOG_Y_MAX - LOG_Y_MIN, 1e-9)
        return oy + plotH - t * plotH
      }

      ctx.fillStyle =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--bg')
          .trim() || '#fff'
      ctx.fillRect(0, 0, W, H)

      ctx.strokeStyle =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--border')
          .trim() || '#ccc'
      ctx.lineWidth = dpr
      ctx.strokeRect(ox, oy, plotW, plotH)

      const gridColor =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--text')
          .trim() || '#888'
      ctx.strokeStyle = gridColor
      ctx.globalAlpha = 0.2
      ctx.lineWidth = 0.5 * dpr
      for (let g = 1; g <= 4; g++) {
        const gx = ox + (g / 5) * plotW
        ctx.beginPath()
        ctx.moveTo(gx, oy)
        ctx.lineTo(gx, oy + plotH)
        ctx.stroke()
      }
      for (const v of Y_AXIS_TICK_BYTES) {
        if (v <= minY || v >= maxY) continue
        const gy = yToPx(v)
        ctx.beginPath()
        ctx.moveTo(ox, gy)
        ctx.lineTo(ox + plotW, gy)
        ctx.stroke()
      }
      ctx.globalAlpha = 1

      const gpuClamped = Math.min(Math.max(gpuBytes, minY), maxY)
      const yGpu = yToPx(gpuClamped)
      const kvOverGpu = currentGqaKvBytes > gpuBytes
      const capStroke =
        kvOverGpu
          ? '#f59e0b'
          : getComputedStyle(document.documentElement)
              .getPropertyValue('--accent')
              .trim() || '#aa3bff'
      ctx.strokeStyle = capStroke
      ctx.globalAlpha = 0.95
      ctx.lineWidth = 1.5 * dpr
      ctx.setLineDash([6 * dpr, 5 * dpr])
      ctx.beginPath()
      ctx.moveTo(ox, yGpu)
      ctx.lineTo(ox + plotW, yGpu)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1

      ctx.fillStyle =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--text-h')
          .trim() || '#111'
      ctx.font = `${10 * dpr}px ui-monospace, monospace`
      ctx.textAlign = 'right'
      const capLabel = `Max memory ~${gpuMemoryGb} GiB`
      if (kvOverGpu) {
        /* KV above budget: keep label under the line so it stays clear of the dot. */
        ctx.textBaseline = 'top'
        ctx.fillText(
          capLabel,
          ox + plotW,
          Math.min(oy + plotH - 4 * dpr, yGpu + 4 * dpr),
        )
      } else {
        ctx.textBaseline = 'bottom'
        ctx.fillText(
          capLabel,
          ox + plotW,
          Math.max(oy + 12 * dpr, yGpu - 4 * dpr),
        )
      }

      const kvLineGqa =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--accent')
          .trim() || '#6366f1'
      const kvLineDense = '#94a3b8'

      const drawKvCurve = (atS, stroke, dash, width) => {
        ctx.strokeStyle = stroke
        ctx.lineWidth = width * dpr
        ctx.setLineDash(dash.map((v) => v * dpr))
        let prevPx = 0
        let prevPy = 0
        let prevKv = 0
        for (let i = 0; i <= KV_CURVE_STEPS; i++) {
          const u = i / KV_CURVE_STEPS
          const s = S_AXIS_MIN * (S_AXIS_MAX / S_AXIS_MIN) ** u
          const kv = atS(s)
          const px = xToPx(s)
          const py = yToPx(kv)
          if (i === 0) {
            prevPx = px
            prevPy = py
            prevKv = kv
            continue
          }
          const bothAbove = prevKv > maxY && kv > maxY
          const bothBelow = prevKv < minY && kv < minY
          if (!bothAbove && !bothBelow) {
            ctx.beginPath()
            ctx.moveTo(prevPx, prevPy)
            ctx.lineTo(px, py)
            ctx.stroke()
          }
          prevPx = px
          prevPy = py
          prevKv = kv
        }
        ctx.setLineDash([])
      }

      drawKvCurve(
        (s) =>
          kvDenseAtS(batch, s, layers, hiddenDim, bytesPerElement),
        kvLineDense,
        [5, 4],
        1.5,
      )
      drawKvCurve(
        (s) =>
          kvGqaAtS(batch, s, layers, numKvHeads, headDim, bytesPerElement),
        kvLineGqa,
        [],
        2,
      )

      const cx = xToPx(
        Math.min(S_AXIS_MAX, Math.max(S_AXIS_MIN, currentSequenceLength)),
      )
      const cy = yToPx(currentGqaKvBytes)
      ctx.fillStyle =
        currentGqaKvBytes > gpuBytes ? '#ef4444' : kvLineGqa
      ctx.beginPath()
      ctx.arc(cx, cy, 4 * dpr, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1 * dpr
      ctx.stroke()

      ctx.fillStyle =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--text')
          .trim() || '#666'
      ctx.font = `${9 * dpr}px ui-monospace, monospace`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      for (const s of X_AXIS_TICK_TOKENS) {
        if (s < S_AXIS_MIN || s > S_AXIS_MAX) continue
        const tx = xToPx(s)
        ctx.fillText(formatXAxisTick(s), tx, oy + plotH + 6 * dpr)
      }
      ctx.fillText(
        'Sequence length S (tokens, log₁₀ scale)',
        ox + plotW / 2,
        oy + plotH + 22 * dpr,
      )

      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      const tickLen = 5 * dpr
      ctx.strokeStyle = gridColor
      ctx.globalAlpha = 0.85
      ctx.lineWidth = 1 * dpr
      ctx.fillStyle =
        getComputedStyle(document.documentElement)
          .getPropertyValue('--text')
          .trim() || '#666'
      ctx.font = `${9 * dpr}px ui-monospace, monospace`
      for (const v of Y_AXIS_TICK_BYTES) {
        const gy = yToPx(v)
        ctx.beginPath()
        ctx.moveTo(ox - tickLen, gy)
        ctx.lineTo(ox, gy)
        ctx.stroke()
        ctx.fillText(formatYAxisTick(v), ox - 12 * dpr, gy)
      }
      ctx.globalAlpha = 1
      ctx.save()
      ctx.translate(padYTitle / 2, oy + plotH / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('KV cache size', 0, 0)
      ctx.restore()

      layoutRef.current = {
        ox,
        oy,
        plotW,
        plotH,
        logSLo: LOG_S_MIN,
        logSHi: LOG_S_MAX,
      }
    }

    const schedulePaint = () => {
      cancelAnimationFrame(resizeRaf)
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = 0
        paint()
      })
    }

    schedulePaint()
    const ro = new ResizeObserver(schedulePaint)
    ro.observe(wrap)
    return () => {
      ro.disconnect()
      cancelAnimationFrame(resizeRaf)
    }
  }, [
    batch,
    layers,
    hiddenDim,
    numKvHeads,
    headDim,
    bytesPerElement,
    gpuBytes,
    gpuMemoryGb,
    currentSequenceLength,
    currentDenseKvBytes,
    currentGqaKvBytes,
  ])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let active = false

    const deviceX = (clientX) => {
      const r = canvas.getBoundingClientRect()
      return ((clientX - r.left) * canvas.width) / Math.max(r.width, 1)
    }

    const deviceY = (clientY) => {
      const r = canvas.getBoundingClientRect()
      return ((clientY - r.top) * canvas.height) / Math.max(r.height, 1)
    }

    const applyFromClientX = (clientX) => {
      const setIn = onInputRef.current
      const setOut = onOutputRef.current
      if (!setIn || !setOut) return
      const L = layoutRef.current
      if (!L) return
      const xd = deviceX(clientX)
      const t = (xd - L.ox) / L.plotW
      if (t < 0 || t > 1) return
      const logS = L.logSLo + t * (L.logSHi - L.logSLo)
      const sChart = Math.round(10 ** logS)
      const inCur = inputTokensRef.current
      const outCur = outputTokensRef.current
      const { in: inNew, out: outNew } = allocateTokensForTotalS(
        inCur,
        outCur,
        sChart,
      )
      setIn(inNew)
      setOut(outNew)
    }

    const onPointerDown = (e) => {
      if (e.button !== 0) return
      const L = layoutRef.current
      if (!L) return
      const xd = deviceX(e.clientX)
      const yd = deviceY(e.clientY)
      if (
        xd < L.ox ||
        xd > L.ox + L.plotW ||
        yd < L.oy ||
        yd > L.oy + L.plotH
      ) {
        return
      }
      active = true
      setDragging(true)
      canvas.setPointerCapture(e.pointerId)
      applyFromClientX(e.clientX)
    }

    const onPointerMove = (e) => {
      if (!active) return
      applyFromClientX(e.clientX)
    }

    const onPointerUp = (e) => {
      if (!active) return
      active = false
      setDragging(false)
      try {
        canvas.releasePointerCapture(e.pointerId)
      } catch {
        /* ignore */
      }
    }

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointercancel', onPointerUp)

    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointercancel', onPointerUp)
    }
  }, [])

  return (
    <div className="kv-seq-chart">
      <div className="kv-seq-chart-head">
        <span>KV vs sequence length</span>
        <span className="mono kv-seq-chart-sub">
          B={batch} · {gpuName} · max memory ~{gpuMemoryGb} GiB
        </span>
      </div>
      <div ref={wrapRef} className="kv-seq-chart-wrap">
        <canvas
          ref={canvasRef}
          className={dragging ? 'kv-chart-dragging' : undefined}
          aria-label="KV vs sequence length; drag horizontally to adjust input and output tokens toward a total sequence length"
        />
      </div>
      <p className="kv-seq-chart-foot">
        Both axes use log₁₀: KV from {formatYAxisTick(Y_AXIS_MIN_BYTES)} to{' '}
        {formatYAxisTick(Y_AXIS_MAX_BYTES)};{' '}
        <span className="mono">S</span> from {S_AXIS_MIN} to {S_AXIS_MAX.toLocaleString()} tokens. Line = KV for
        your batch, model, and dtype. Dotted line = max memory for the selected GPU; it turns
        amber and the label moves below the line when KV exceeds that budget. Dot = current S = input + output.
        {onOutputTokensChange && onInputTokensChange ? (
          <>
            {' '}
            <strong>Drag</strong> horizontally to set total <span className="mono">S</span>:
            output moves first; at {TOK_MIN} output, input moves down to {TOK_MIN}. Increasing{' '}
            <span className="mono">S</span> fills output first, then input.
          </>
        ) : null}
      </p>
    </div>
  )
}
