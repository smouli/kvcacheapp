/** Integer powers of 2: 2^minExp … 2^maxExp (inclusive). */
function powersOfTwoRange(minExp, maxExp) {
  const out = []
  for (let e = minExp; e <= maxExp; e++) {
    out.push(2 ** e)
  }
  return out
}

/** Token dropdowns (prompt / max-new): 2^7 … 2^17 (128 … 131072). */
export const TOKEN_OPTIONS = powersOfTwoRange(7, 17)

/** Batch axis: 2^0 … 2^8 (1 … 256). */
export const BATCH_OPTIONS = powersOfTwoRange(0, 8)

/** Sequence length S: 2^8 … 2^17 (256 … 131072). */
export const S_OPTIONS = powersOfTwoRange(8, 17)

export function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}

/** Nearest value in a sorted ascending array. */
export function nearestIn(sortedArr, x) {
  let best = sortedArr[0]
  let bestD = Math.abs(x - best)
  for (const v of sortedArr) {
    const d = Math.abs(x - v)
    if (d < bestD) {
      bestD = d
      best = v
    }
  }
  return best
}

export function batchToRowIndex(batch) {
  const nearest = nearestIn(BATCH_OPTIONS, batch)
  return BATCH_OPTIONS.indexOf(nearest)
}

export function sToColIndex(S) {
  const nearest = nearestIn(S_OPTIONS, S)
  return S_OPTIONS.indexOf(nearest)
}

/**
 * Pick (input, output) from TOKEN_OPTIONS whose sum is closest to targetS,
 * preferring pairs near the previous split.
 */
export function syncTokensToSum(targetS, prevInput, prevOutput) {
  const S = Math.max(2, Math.round(targetS))
  let best = { input: TOKEN_OPTIONS[0], output: TOKEN_OPTIONS[0] }
  let bestScore = Infinity
  for (const i of TOKEN_OPTIONS) {
    for (const o of TOKEN_OPTIONS) {
      if (i + o !== S) continue
      const score =
        Math.abs(i - prevInput) +
        Math.abs(o - prevOutput) +
        Math.abs(i / S - prevInput / Math.max(prevInput + prevOutput, 1)) * 0.001
      if (score < bestScore) {
        bestScore = score
        best = { input: i, output: o }
      }
    }
  }
  if (bestScore < Infinity) return best

  let pair = { input: prevInput, output: prevOutput }
  let bestSumD = Infinity
  for (const i of TOKEN_OPTIONS) {
    for (const o of TOKEN_OPTIONS) {
      const sum = i + o
      const d = Math.abs(sum - S)
      if (d < bestSumD) {
        bestSumD = d
        pair = { input: i, output: o }
      }
    }
  }
  return pair
}
