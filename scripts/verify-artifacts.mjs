import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const catalog = JSON.parse(await fs.readFile(path.join(ROOT, "catalog.json")));

async function verifyBytes(artifact) {
  const response = await fetch(artifact.url, {
    redirect: "follow",
    headers: { Origin: "https://example.invalid" },
  });
  if (!response.ok) throw new Error(`${artifact.url} -> HTTP ${response.status}`);
  const hash = crypto.createHash("sha256");
  let bytes = 0;
  for await (const chunk of response.body) {
    hash.update(chunk);
    bytes += chunk.byteLength;
  }
  const sha256 = hash.digest("hex");
  if (bytes !== artifact.bytes) {
    throw new Error(`${artifact.name}: ${bytes} bytes, expected ${artifact.bytes}`);
  }
  if (sha256 !== artifact.sha256) {
    throw new Error(`${artifact.name}: ${sha256}, expected ${artifact.sha256}`);
  }
  console.log(`  ok  ${artifact.name} ${bytes} bytes ${sha256}`);
}

for (const family of Object.values(catalog.families)) {
  for (const row of family.entries.filter((entry) => entry.runtimeKind === "webllm")) {
    const entryDir = path.join(ROOT, row.path);
    const entry = JSON.parse(await fs.readFile(path.join(entryDir, "entry.json")));
    const manifest = JSON.parse(
      await fs.readFile(path.join(entryDir, entry.artifacts)),
    );
    console.log(`${entry.family}/${entry.id}`);
    for (const artifact of [...manifest.runtimeJs, ...manifest.modelWasm]) {
      await verifyBytes(artifact);
    }
    for (const resource of manifest.modelResources) {
      if (resource.revision !== entry.model.revision) {
        throw new Error(`${resource.name}: revision does not match entry.model`);
      }
      const probe = new URL("mlc-chat-config.json", resource.url);
      const response = await fetch(probe, {
        redirect: "follow",
        headers: { Origin: "https://example.invalid" },
      });
      if (!response.ok) throw new Error(`${probe} -> HTTP ${response.status}`);
      if (response.headers.get("access-control-allow-origin") == null) {
        throw new Error(`${probe}: no CORS access-control-allow-origin header`);
      }
      console.log(`  ok  ${resource.name}@${resource.revision}`);
    }
  }
}
