export async function sha256(buf: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,"0")).join("");
}

export async function detectEnv(): Promise<import("./types").EnvInfo> {
  const webgpu = !!(navigator as any).gpu;
  let approxVramMB = 0;
  if (webgpu) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      // very rough heuristic using limits
      const lim = adapter?.limits;
      if (lim?.maxStorageBufferBindingSize) {
        approxVramMB = Math.round((lim.maxStorageBufferBindingSize / 1048576) * 1.5);
      }
    } catch {}
  }
  // WASM support baseline
  const wasm = typeof WebAssembly !== "undefined";
  // light SIMDish probe (don’t block)
  const wasmSimd = wasm && typeof (WebAssembly as any).validate === "function";
  const threads = (navigator as any).hardwareConcurrency || 4;
  return { webgpu, wasm, wasmSimd, threads, approxVramMB };
}