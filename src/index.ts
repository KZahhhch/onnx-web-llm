// src/index.ts
import * as ort from "onnxruntime-web";
import { AutoTokenizer, env as xenovaEnv } from "@xenova/transformers";

export interface InitOptions {
  baseURL?: string; // prefix for model + tokenizer files
  executionProviders?: ("webgpu" | "wasm")[];
  tokenizerPath?: string; // e.g., "tokenizer/" (must contain tokenizer.json etc.)
}

export class OnnxWebLLM {
  private session?: ort.InferenceSession;
  private tokenizer?: any; // AutoTokenizer
  private opts: InitOptions;

  constructor(opts: InitOptions) {
    this.opts = opts;
    // Optional: point Transformers.js to your CDN root for any relative fetches
    if (this.opts.baseURL) xenovaEnv.localModelPath = this.opts.baseURL;
  }

  async initTokenizer() {
    if (!this.opts.tokenizerPath) throw new Error("tokenizerPath is required");
    // Loads tokenizer.json from `${baseURL}/${tokenizerPath}`
    this.tokenizer = await AutoTokenizer.from_pretrained(this.opts.tokenizerPath);
  }

  async loadBase(modelPath: string) {
    this.session = await ort.InferenceSession.create(
      this.withBase(modelPath),
      { executionProviders: this.opts.executionProviders ?? ["webgpu", "wasm"] }
    );
  }

  // Minimal demo generate (stubbed sampling / KV-cache for now)
  async *generate(prompt: string) {
    if (!this.session) throw new Error("Model not loaded");
    if (!this.tokenizer) await this.initTokenizer();

    const enc = await this.tokenizer.encode(prompt);
    // TODO: wire your model's specific IO names + KV-cache and sampling loop.
    // For now, just prove the pipeline works:
    yield { text: "Hello from onnx-web-llm (Transformers.js tokenizer)" };
  }

  private withBase(path: string) {
    if (!this.opts.baseURL) return path;
    return path.startsWith("http") ? path : `${this.opts.baseURL}${path}`;
  }
}

// Example (optional): quick smoke test if you run this module directly in the browser
(async () => {
  // no-op in node; works when imported in a <script type="module">
})();
