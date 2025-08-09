import * as ort from "onnxruntime-web";
import { Provider } from "../types";

export interface Session {
  session: ort.InferenceSession;
}

export class OrtEngine {
  private providers: Provider[] = ["webgpu","wasm"];

  async init(providers?: Provider[]) {
    this.providers = providers ?? this.providers;
    try {
      // init WebGPU if present; ignore if not available
      await (ort as any).env.webgpu?.init?.();
    } catch {}
    (ort as any).env.wasm.numThreads = (navigator as any).hardwareConcurrency || 4;
    (ort as any).env.wasm.simd = true;
  }

  async createSession(modelBuf: ArrayBuffer): Promise<Session> {
    const session = await ort.InferenceSession.create(modelBuf, {
      executionProviders: this.providers as any,
    });
    return { session };
  }

  // placeholder: wire once .onnx_adapter APIs are exposed in web
  async applyAdapter(_sess: Session, _adapterBuf: ArrayBuffer): Promise<void> {
    // TODO: implement when public adapter attach API is available to JS
    // For now, keep method signature so SDK surface is stable.
    throw new Error("applyAdapter not implemented yet");
  }

  async run(sess: Session, feeds: Record<string, ort.Tensor>): Promise<Record<string, ort.Tensor>> {
    return await sess.session.run(feeds);
  }
}