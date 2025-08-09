import { loadManifest } from "./manifest";
import { selectVariant } from "./selector";
import { detectEnv } from "./utils";
import { fetchHubFile } from "./fetchers/hub";
import { OrtEngine, Session } from "./engine/ort-web";
import type { AdapterEntry, BaseId, Manifest, Tokenizer, VariantEntry } from "./types";

export interface InitOptions {
  manifestURL: string;
  providers?: ("webgpu" | "wasm")[];
  tokenizer: Tokenizer;                 // keep tokenizer pluggable
  hubToken?: string;                    // optional HF token
}

export class OnnxWebLLM {
  private opts: InitOptions;
  private manifest!: Manifest;
  private engine = new OrtEngine();
  private session?: Session;
  private activeBase?: BaseId;
  private activeVariant?: VariantEntry;
  private adapters = new Map<string, AdapterEntry>();
  private tokenizer: Tokenizer;

  constructor(opts: InitOptions) {
    this.opts = opts;
    this.tokenizer = opts.tokenizer;
  }

  async init() {
    this.manifest = await loadManifest(this.opts.manifestURL);
    await this.engine.init(this.opts.providers);
  }

  async useBase(baseId: BaseId, overrideVariantId?: string) {
    const base = this.manifest.bases.find(b => b.id === baseId);
    if (!base) throw new Error(`Base not found in manifest: ${baseId}`);

    const env = await detectEnv();
    const variant = overrideVariantId
      ? (base.variants.find(v => v.id === overrideVariantId) ?? base.variants[0])
      : selectVariant(env, base.variants);

    const modelBuf = await fetchHubFile(variant.repo, variant.path, variant.rev ?? "main", {
      token: this.opts.hubToken, cacheName: "onnx-web-llm"
    });

    this.session = await this.engine.createSession(modelBuf);
    this.activeBase = baseId;
    this.activeVariant = variant;
  }

  async registerAdapter(entry: AdapterEntry) {
    if (this.activeBase && entry.base !== this.activeBase) {
      throw new Error(`Adapter ${entry.id} is for base ${entry.base}, but active base is ${this.activeBase}`);
    }
    this.adapters.set(entry.id, entry);
  }

  async useAdapter(adapterId: string | null) {
    if (!this.session) throw new Error("No active session");
    if (adapterId === null) return; // base-only

    const entry = this.adapters.get(adapterId);
    if (!entry) throw new Error(`Unknown adapter: ${adapterId}`);
    const buf = await fetch(entry.url).then(r=>r.arrayBuffer());
    await this.engine.applyAdapter(this.session, buf); // will throw until implemented
  }

  // Minimal streaming loop stub (you'll wire IO names for Phi-3 later)
  async *generate(opts: { prompt?: string; messages?: {role:"system"|"user"|"assistant";content:string}[]; maxTokens?: number; temperature?: number; }) {
    if (!this.session) throw new Error("No active session");
    const prompt = opts.prompt ?? (opts.messages ? this.renderChat(opts.messages) : "");
    const ids = await this.tokenizer.encode(prompt);
    // TODO: feed ids to ONNX graph, manage KV cache, sample next tokens
    // For now, emit a placeholder delta so the DX is in place:
    yield { text: "(generation not yet wired)" };
  }

  private renderChat(msgs: {role:"system"|"user"|"assistant";content:string}[]) {
    // simple template; customize per base later
    return msgs.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n") + "\nASSISTANT: ";
  }

  get activeModelInfo() {
    return { base: this.activeBase, variant: this.activeVariant?.id };
  }
}