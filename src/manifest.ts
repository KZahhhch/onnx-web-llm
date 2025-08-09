import { Manifest } from "./types";

export async function loadManifest(url: string): Promise<Manifest> {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`);
  const json = await res.json();
  // minimal validation
  if (!json?.version || !Array.isArray(json?.bases)) {
    throw new Error("Invalid manifest format");
  }
  return json as Manifest;
}