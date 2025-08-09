import { EnvInfo, VariantEntry } from "./types";

export function selectVariant(env: EnvInfo, variants: VariantEntry[]): VariantEntry {
  const viable = variants.filter(v => {
    const epOk = env.webgpu ? v.providers.includes("webgpu") : v.providers.includes("wasm");
    const memOk = (v.min_vram_mb ?? 0) <= (env.approxVramMB || 0) || !v.min_vram_mb;
    return epOk && memOk;
  });
  const score = (v: VariantEntry) => {
    let s = 0;
    if (v.providers.includes("webgpu") && env.webgpu) s += 100;
    if (v.precision === "fp16") s += 6;
    if (v.precision === "int8") s += 4;
    if (v.precision === "int4") s += 2;
    if ((v.min_vram_mb ?? 0) <= (env.approxVramMB || 0)) s += 5;
    return s;
  };
  return (viable.length ? viable : variants).sort((a,b)=>score(b)-score(a))[0];
}