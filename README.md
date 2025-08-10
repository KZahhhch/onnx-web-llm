# onnx-web-llm

A browser-first, **manifest-driven** SDK for running ONNX-exported LLMs entirely on the client (WebGPU/WASM), with hardware-aware model variant selection and a public API designed for future **LoRA adapter swapping**.

---

## Overview

This repo is the foundation for a WebLLM-like developer experience built on **ONNX Runtime Web**. The design goals:

- **Future-proof model selection** via a remote, versioned **manifest** (add/retire models without changing SDK code).
- **Hardware-aware** runtime choice (WebGPU vs WASM) and model **variant** selection (e.g., FP16 vs INT4) at load time.
- **Hugging Face Hub** integration with caching and optional SHA integrity checks.
- **Tokenizer-agnostic** interface (works with `@xenova/transformers` or any compatible tokenizer).
- **Adapter-ready** API surface (`registerAdapter`, `useAdapter`) for LoRA hot-swap once the browser attach API is available.
- **Engine abstraction** around ONNX Runtime Web so SDK surface remains stable if internals change.

This README explains what’s implemented so far and how the pieces fit together.

---

## What’s in place (high level)

- **Manifest loader & selector**
  - Loads a JSON manifest describing bases and their variants.
  - Detects the browser’s capabilities (WebGPU, rough VRAM heuristic, WASM SIMD, threads).
  - Selects the best variant for the device, with an override option.

- **Hub fetcher with cache + optional SHA**
  - Downloads model files from the **Hugging Face Hub** using resolve URLs.
  - Stores artifacts in **Cache Storage**; validates content via **SHA-256** if provided.

- **ONNX Runtime Web engine wrapper**
  - Small class that initializes ORT Web (WebGPU/WASM), creates sessions from ArrayBuffers, and exposes a `run` call.
  - Includes a stubbed `applyAdapter` method so the public API won’t change when LoRA attach is plugged in.

- **Public SDK surface**
  - `OnnxWebLLM` with `init()`, `useBase()`, `registerAdapter()`, `useAdapter()`, and `generate()` (streaming stub).
  - Tokenizer abstraction with a minimal interface: `encode(text)`, `decode(ids)`.

- **Example manifest**
  - Describes **Phi-3 mini 4k** base with multiple browser-suitable variants (e.g., `webgpu-fp16`, `webgpu-int4`, `wasm-int4`).

---

## Project structure

    src/
      types.ts           # Shared types (manifest, env info, providers, tokenizer interface)
      utils.ts           # SHA-256 helper, environment detection
      manifest.ts        # Manifest fetch + minimal validation
      selector.ts        # Variant selection logic (scores by provider/precision/memory)
      fetchers/
        hub.ts           # Hub "resolve" fetch + Cache Storage + optional SHA verify
      engine/
        ort-web.ts       # ONNX Runtime Web wrapper (providers init, createSession, run, applyAdapter stub)
      index.ts           # Public SDK (OnnxWebLLM): init/useBase/registerAdapter/useAdapter/generate
    public/
      manifest.json      # Example manifest (can be hosted anywhere)

---

## The manifest (how models are described)

The SDK relies on a versioned JSON manifest. This keeps “which models to ship” out of SDK code and lets you roll updates safely. You can serve different manifests to different deployments.

Example:

    {
      "version": "2025.08.01",
      "bases": [
        {
          "id": "phi3-mini-4k",
          "tokenizer": { "repo": "microsoft/Phi-3-mini-4k-instruct" },
          "variants": [
            {
              "id": "webgpu-fp16",
              "repo": "microsoft/Phi-3-mini-4k-instruct-onnx-web",
              "path": "webgpu/fp16/model.onnx",
              "precision": "fp16",
              "providers": ["webgpu"],
              "min_vram_mb": 3000,
              "max_seq_len": 4096
            },
            {
              "id": "webgpu-int4",
              "repo": "microsoft/Phi-3-mini-4k-instruct-onnx-web",
              "path": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx",
              "precision": "int4",
              "providers": ["webgpu","wasm"],
              "min_vram_mb": 700,
              "max_seq_len": 4096
            },
            {
              "id": "wasm-int4",
              "repo": "microsoft/Phi-3-mini-4k-instruct-onnx-web",
              "path": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx",
              "precision": "int4",
              "providers": ["wasm"],
              "max_seq_len": 4096
            }
          ]
        }
      ]
    }

Key ideas:
- **bases**: logical model families (e.g., `phi3-mini-4k`, `llama3-8b`).
- **variants**: concrete artifacts tuned for a provider/precision (WebGPU FP16 vs WASM INT4, etc.).
- You can add `sha256` to enforce integrity, and `graph_signature` to ensure adapter compatibility.

---

## Environment detection & variant selection

At load time, the SDK:
- Checks **WebGPU** availability and reads limits (rough **VRAM** heuristic).
- Checks **WASM** and a light **SIMD** signal.
- Reads **hardwareConcurrency** for threading hints.

A simple, testable selector function ranks variants (prefer WebGPU, then higher precision if memory allows, otherwise WASM). Apps can override the choice by ID for debugging or A/B experiments.

---

## Hugging Face Hub integration (auto-download)

The SDK downloads artifacts at runtime using Hub **resolve** URLs and caches them with **Cache Storage**. Optional **SHA-256** verification prevents accidental mismatches. You may provide an HF **token** for gated models.

No local `public/model.onnx` is required—artifacts stream straight from the Hub and are cached in the browser.

---

## Tokenizer abstraction

We keep tokenization decoupled:

- Any tokenizer object that implements:
  
      encode(text: string): Promise<Uint32Array>
      decode(ids: Uint32Array): Promise<string>

  can be used.

- A common choice is `@xenova/transformers`:
  
      import { AutoTokenizer } from "@xenova/transformers";
      const tok = await AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct");
      const tokenizer = {
        encode: async t => new Uint32Array((await tok.encode(t)).ids),
        decode: async ids => tok.decode([...ids]),
      };

---

## Public SDK surface (today)

    import { OnnxWebLLM } from "./src/index";
    // tokenizer implements the Tokenizer interface
    const llm = new OnnxWebLLM({
      manifestURL: "/manifest.json",
      providers: ["webgpu","wasm"],
      tokenizer,
      hubToken: undefined, // optional
    });

    await llm.init();
    await llm.useBase("phi3-mini-4k");  // auto-selects best variant
    console.log(llm.activeModelInfo);   // { base, variant }

    await llm.registerAdapter({
      id: "finance-v1",
      base: "phi3-mini-4k",
      url: "https://cdn.example.com/adapters/finance.onnx_adapter"
    });

    await llm.useAdapter("finance-v1"); // no-op until adapter attach is wired in Web

    for await (const chunk of llm.generate({ prompt: "Hello!" })) {
      console.log(chunk.text); // currently a placeholder; KV-cache loop is next
    }

Notes:
- `generate()` currently stubs streaming to lock in the developer experience and API shape. Next step wires the Phi-3 ONNX I/O names and KV-cache for real tokens.
- `useAdapter()` calls the engine’s `applyAdapter()` which is intentionally a stub for now. When ONNX Runtime Web exposes the LoRA attach API, we’ll implement it without changing the SDK surface.

---

## Why this approach is future-proof

- **Model choice is data, not code**: add new bases/variants by updating the manifest.
- **Stable public API**: we already expose `registerAdapter`/`useAdapter` so LoRA hot-swap won’t require a breaking release.
- **Pluggable internals**: ONNX Runtime Web is wrapped; tokenizer is an interface; fetching/caching is modular.
- **Integrity & compatibility**: optional `sha256` and `graph_signature` checks can be enforced to prevent subtle mismatches.

---

## What’s next

- Implement the **Phi-3** ONNX I/O wiring and **KV-cache** loop for real streaming tokens.
- Add a **Service Worker** to pre-cache model/tokenizer and support offline use.
- Wire **LoRA adapter attach** once ONNX Runtime Web surfaces the adapter API in JS.
- Telemetry hooks: `onModelChosen`, `onPerf` (TTFT, tok/s), `onError`.

---

## License

MIT
