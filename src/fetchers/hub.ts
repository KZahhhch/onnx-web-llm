import { sha256 } from "../utils";

const HF = "https://huggingface.co";

export interface HubOpts {
  token?: string;   // optional HF token
  cacheName?: string; // Cache Storage bucket
}

export async function fetchHubFile(
  repo: string,
  path: string,
  rev = "main",
  opts: HubOpts = {}
): Promise<ArrayBuffer> {
  const url = `${HF}/${repo}/resolve/${rev}/${path}?download=1`;
  return fetchWithCache(url, opts);
}

export async function fetchWithCache(url: string, opts: HubOpts = {}, expectedSha?: string): Promise<ArrayBuffer> {
  const cacheName = opts.cacheName ?? "onnx-web-llm";
  const cache = "caches" in self ? await caches.open(cacheName) : null;

  // try cache first
  if (cache) {
    const hit = await cache.match(url);
    if (hit && hit.ok) {
      const buf = await hit.arrayBuffer();
      if (!expectedSha || await sha256(buf) === expectedSha) return buf;
      // sha mismatch - fall through to re-fetch
    }
  }

  const headers: Record<string,string> = {};
  if (opts.token) headers["Authorization"] = `Bearer ${opts.token}`;

  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`Fetch failed ${res.status}: ${url}`);
  const buf = await res.arrayBuffer();

  if (expectedSha) {
    const got = await sha256(buf);
    if (got !== expectedSha) throw new Error(`SHA mismatch for ${url}. expected=${expectedSha} got=${got}`);
  }

  if (cache) {
    const resp = new Response(buf, { headers: { "Content-Type": "application/octet-stream" }});
    await cache.put(url, resp);
  }
  return buf;
}