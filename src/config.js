/**
 * OSS model presets for KV sizing.
 * Qwen2.5-7B uses GQA (28 Q heads, 4 KV heads); others use representative shapes.
 */
export const MODELS = {
  qwen: {
    id: 'qwen',
    name: 'Qwen2.5-7B',
    hfId: 'Qwen/Qwen2.5-7B-Instruct',
    blurb: '28 layers · GQA 28→4 heads',
    layers: 28,
    hiddenDim: 3584,
    numAttentionHeads: 28,
    numKvHeads: 4,
    headDim: 128,
  },
  kimi: {
    id: 'kimi',
    name: 'Kimi (Moonshot)',
    hfId: 'moonshotai/Kimi-K2-Instruct',
    blurb: '36 layers · GQA-class',
    layers: 36,
    hiddenDim: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
  },
  glm: {
    id: 'glm',
    name: 'GLM-4-9B',
    hfId: 'THUDM/glm-4-9b-chat',
    blurb: '40 layers · GQA-class',
    layers: 40,
    hiddenDim: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
  },
  nemotron: {
    id: 'nemotron',
    name: 'Nemotron 8B',
    hfId: 'nvidia/Nemotron-Mini-4B-Instruct',
    blurb: '32 layers · GQA-class',
    layers: 32,
    hiddenDim: 4096,
    numAttentionHeads: 32,
    numKvHeads: 8,
    headDim: 128,
  },
  gptOss: {
    id: 'gptOss',
    name: 'gpt-oss ~20B',
    hfId: 'openai/gpt-oss-20b',
    blurb: '24 layers · mid-width',
    layers: 24,
    hiddenDim: 2304,
    numAttentionHeads: 18,
    numKvHeads: 6,
    headDim: 128,
  },
}

/** Datacenter & cloud GPU SKUs (per accelerator). */
export const GPUS = {
  doH100: {
    id: 'doH100',
    name: 'DO GPU H100',
    detail: '80 GB · DigitalOcean droplet',
    memoryGb: 80,
  },
  a100: { id: 'a100', name: 'A100', detail: '80 GB HBM2e', memoryGb: 80 },
  h100: { id: 'h100', name: 'H100', detail: '80 GB HBM3', memoryGb: 80 },
  h200: { id: 'h200', name: 'H200', detail: '141 GB HBM3e', memoryGb: 141 },
  b200: { id: 'b200', name: 'B200', detail: '~192 GB HBM3e', memoryGb: 192 },
}

export const PRECISION = {
  fp32: { id: 'fp32', label: 'FP32', bytesPerElement: 4 },
  fp16: { id: 'fp16', label: 'FP16', bytesPerElement: 2 },
  bf16: { id: 'bf16', label: 'BF16', bytesPerElement: 2 },
  fp8: { id: 'fp8', label: 'FP8', bytesPerElement: 1 },
  int8: { id: 'int8', label: 'INT8', bytesPerElement: 1 },
}
