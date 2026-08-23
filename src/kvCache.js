/**
 * Dense MHA KV: batch × S × L × 2 × hidden_dim × bytes
 * GQA KV:      batch × S × L × 2 × num_kv_heads × head_dim × bytes
 */

export function computeDenseKvCacheBytes({
  batch,
  sequenceLength,
  layers,
  hiddenDim,
  bytesPerElement,
}) {
  return (
    batch * sequenceLength * layers * 2 * hiddenDim * bytesPerElement
  )
}

/** @deprecated alias */
export function computeKvCacheBytes(params) {
  return computeDenseKvCacheBytes(params)
}

export function computeGqaKvCacheBytes({
  batch,
  sequenceLength,
  layers,
  numKvHeads,
  headDim,
  bytesPerElement,
}) {
  return (
    batch * sequenceLength * layers * 2 * numKvHeads * headDim * bytesPerElement
  )
}

export function headDimFromModel(model) {
  if (model.headDim) return model.headDim
  if (model.numAttentionHeads) {
    return Math.round(model.hiddenDim / model.numAttentionHeads)
  }
  return model.hiddenDim
}

export function kvReductionRatio(model) {
  const heads = model.numAttentionHeads ?? model.numKvHeads ?? 1
  const kv = model.numKvHeads ?? heads
  return heads / Math.max(kv, 1)
}

export function formatBytes(bytes) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  const i = Math.min(
    sizes.length - 1,
    Math.floor(Math.log(bytes) / Math.log(k)),
  )
  const value = bytes / k ** i
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2
  return `${value.toFixed(digits)} ${sizes[i]}`
}

export function bytesToGiB(bytes) {
  return bytes / 1024 ** 3
}
