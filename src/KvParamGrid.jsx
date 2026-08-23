import { Fragment, useCallback, useMemo, useRef } from 'react'
import { computeKvCacheBytes } from './kvCache'
import {
  BATCH_OPTIONS,
  S_OPTIONS,
  batchToRowIndex,
  clamp,
  sToColIndex,
} from './kvGridUtils'

/**
 * Map normalized pointer (0→1 along axis) to a cell index for a uniform
 * CSS grid: equal cell sizes, so hit-testing is linear in position (axis
 * values are powers of 2).
 */
function normToUniformCellIndex(norm, n) {
  if (n <= 1) return 0
  const u = clamp(norm, 0, 1)
  return Math.min(n - 1, Math.max(0, Math.floor(u * n)))
}

export function KvParamGrid({
  batch,
  sequenceLength,
  inputTokens,
  outputTokens,
  onPick,
  layers,
  hiddenDim,
  bytesPerElement,
  gpuBytes,
}) {
  const wrapRef = useRef(null)

  const nRows = BATCH_OPTIONS.length
  const nCols = S_OPTIONS.length

  const rowIndex = useMemo(() => batchToRowIndex(batch), [batch])
  const colIndex = useMemo(() => sToColIndex(sequenceLength), [sequenceLength])

  const colTemplate = useMemo(
    () => `repeat(${nCols}, minmax(14px, 1fr))`,
    [nCols],
  )

  const cellKv = useCallback(
    (b, s) =>
      computeKvCacheBytes({
        batch: b,
        sequenceLength: s,
        layers,
        hiddenDim,
        bytesPerElement,
      }),
    [layers, hiddenDim, bytesPerElement],
  )

  const emitFromPointer = useCallback(
    (clientX, clientY) => {
      const el = wrapRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const nx = (clientX - r.left) / Math.max(r.width, 1)
      const ny = (clientY - r.top) / Math.max(r.height, 1)
      const ci = normToUniformCellIndex(nx, nCols)
      const ri = normToUniformCellIndex(ny, nRows)
      const newBatch = BATCH_OPTIONS[ri]
      const newS = S_OPTIONS[ci]
      onPick({ batch: newBatch, sequenceLength: newS })
    },
    [nRows, nCols, onPick],
  )

  const onPointerDown = (e) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    emitFromPointer(e.clientX, e.clientY)
  }

  const onPointerMove = (e) => {
    if ((e.buttons & 1) === 0) return
    emitFromPointer(e.clientX, e.clientY)
  }

  const endDrag = (e) => {
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="kv-grid-block">
      <p className="kv-grid-title">Batch × sequence (powers of 2)</p>
      <p className="kv-grid-lede">
        Drag inside the cells: <strong>right/left</strong> for sequence length{' '}
        <span className="mono">S</span> (powers of 2 along the axis);{' '}
        <strong>down</strong> for larger <span className="mono">B</span>. Dropdowns choose prompt and max-new
        from the same token lists; after a drag, both snap to a pair that sums to the
        column <span className="mono">S</span>.
      </p>

      <div className="kv-scroll-x">
        <div
          className="kv-grid-sheet"
          style={{ minWidth: `${2.25 + nCols * 1.1}rem` }}
        >
          <div className="kv-grid-corner" aria-hidden />
          <div
            className="kv-grid-col-head"
            style={{ gridTemplateColumns: colTemplate }}
          >
            {S_OPTIONS.map((s) => (
              <span key={s} className="kv-grid-head-cell mono" title={`S = ${s}`}>
                {s >= 10000 ? `${Math.round(s / 1000)}k` : s}
              </span>
            ))}
          </div>
          <div
            className="kv-grid-row-head"
            style={{ gridTemplateRows: `repeat(${nRows}, 1fr)` }}
          >
            {BATCH_OPTIONS.map((b) => (
              <span key={b} className="kv-grid-head-cell mono" title={`B = ${b}`}>
                {b}
              </span>
            ))}
          </div>
          <div
            ref={wrapRef}
            className="kv-grid-surface"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            role="application"
            aria-label="Drag to set batch and sequence length"
          >
            <div
              className="kv-grid-cells"
              style={{
                gridTemplateRows: `repeat(${nRows}, 1fr)`,
                gridTemplateColumns: colTemplate,
              }}
            >
              {BATCH_OPTIONS.map((b, ri) => (
                <Fragment key={b}>
                  {S_OPTIONS.map((s, ci) => {
                    const kv = cellKv(b, s)
                    const over = kv > gpuBytes
                    const t = Math.min(
                      1,
                      Math.log1p(kv / Math.max(gpuBytes, 1)) / Math.log(4),
                    )
                    const sel = ri === rowIndex && ci === colIndex
                    return (
                      <div
                        key={`${b}-${s}`}
                        className={`kv-grid-cell ${over ? 'over' : ''} ${sel ? 'selected' : ''}`}
                        style={{
                          opacity: 0.35 + 0.55 * (1 - t * 0.85),
                          background: over
                            ? `color-mix(in srgb, var(--accent) ${Math.round(55 + 40 * t)}%, #ef4444)`
                            : `color-mix(in srgb, var(--accent) ${Math.round(18 + 35 * t)}%, var(--border))`,
                        }}
                        title={`B=${b}, S=${s.toLocaleString()} → ${(kv / 1024 ** 2).toFixed(1)} MiB KV`}
                      />
                    )
                  })}
                </Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>

      <p className="kv-grid-foot mono">
        Selected: B={batch}, S={sequenceLength.toLocaleString()} (input{' '}
        {inputTokens.toLocaleString()} + output {outputTokens.toLocaleString()})
      </p>
    </div>
  )
}
