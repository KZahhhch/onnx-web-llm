export type Provider = "webgpu" | "wasm";

export type BaseId = string;
export type VariantId = string;
export type AdapterId = string;

export interface Manifest {
  version: string;                 // e.g., "2025.08.01"
  bases: BaseEntry[];
}

export interface BaseEntry {
  id: BaseId;                      // e.g., "phi3-mini-4k"
  tokenizer: { repo: string; rev?: string };  // HF repo for tokenizer
  variants: VariantEntry[];
}

export interface VariantEntry {
  id: VariantId;                   // e.g., "webgpu-fp16"
  repo: string;                    // HF repo with ONNX blobs
  path: string;                    // path to model.onnx in that repo
  rev?: string;                    // git ref on HF (default "main")
  precision: "fp16" | "int8" | "int4" | "other";
  providers: Provider[];           // which EPs this variant targets
  min_vram_mb?: number;            // heuristic budget
  max_seq_len?: number;
  sha256?: string;                 // optional integrity check
  graph_signature?: string;        // optional IO signature hash
}

export interface AdapterEntry {
  id: AdapterId;
  base: BaseId;
  url: string;                     // absolute or relative URL (can be HF resolve)
  sha256?: string;
  targets?: string[];              // optional layer names; for validation
  format_version?: string;         // ".onnx_adapter" spec ver
}

export interface EnvInfo {
  webgpu: boolean;
  wasm: boolean;
  wasmSimd: boolean;
  threads: number;
  approxVramMB: number;
}

export interface Tokenizer {
  encode(text: string): Promise<Uint32Array>;
  decode(ids: Uint32Array): Promise<string>;
}