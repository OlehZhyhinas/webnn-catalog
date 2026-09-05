// Serve the catalog plus a weights directory over localhost, so a browser can
// load a recipe without a build step.
//
//   node scripts/serve.mjs [--port 8903] [--weights <dir>]
//
// The catalog is at /, the constants blobs at /weights. Range requests are
// answered, which the loader's chunked constant path requires and the 1.65 GB
// whole-file path benefits from. WebNN is secure-context only, and localhost
// counts, so this is enough: no TLS, no build, no bundler.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DEFAULT_WEIGHTS = path.resolve(ROOT, "..", "webnn-workbench", "bench", "webnn", "ir");

let port = 8903;
let weights = DEFAULT_WEIGHTS;
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === "--port") port = parseInt(argv[++i], 10);
  else if (argv[i] === "--weights") weights = path.resolve(argv[++i]);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".txt": "text/plain; charset=utf-8",
  ".css": "text/css; charset=utf-8",
};

const MOUNTS = [["/weights", weights], ["", ROOT]];

const resolveUrl = (urlPath) => {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  for (const [prefix, dir] of MOUNTS) {
    if (decoded === prefix || decoded.startsWith(prefix + "/")) {
      const rel = decoded.slice(prefix.length).replace(/^\//, "") || "demo/index.html";
      const p = path.join(dir, path.normalize("/" + rel));
      if (!p.startsWith(dir)) return null;
      return p;
    }
  }
  return null;
};

http
  .createServer((req, res) => {
    res.setHeader("Cache-Control", "no-store");
    if (req.url === "/favicon.ico") return void res.writeHead(204).end();
    let filePath = resolveUrl(req.url);
    if (!filePath) return void res.writeHead(404).end(`no mount for ${req.url}`);
    let stat;
    try { stat = fs.statSync(filePath); } catch { return void res.writeHead(404).end(`not found: ${req.url}`); }
    if (stat.isDirectory()) {
      filePath = path.join(filePath, "index.html");
      try { stat = fs.statSync(filePath); } catch { return void res.writeHead(404).end(`no index in ${req.url}`); }
    }
    const type = MIME[path.extname(filePath).toLowerCase()] ?? "application/octet-stream";
    const range = req.headers.range && /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (range) {
      const start = range[1] === "" ? stat.size - Number(range[2]) : Number(range[1]);
      const end = range[1] === "" || range[2] === "" ? stat.size - 1 : Number(range[2]);
      if (start >= 0 && end < stat.size && start <= end) {
        res.writeHead(206, {
          "Content-Type": type,
          "Content-Length": end - start + 1,
          "Content-Range": `bytes ${start}-${end}/${stat.size}`,
          "Accept-Ranges": "bytes",
        });
        return void fs.createReadStream(filePath, { start, end }).pipe(res);
      }
    }
    res.writeHead(200, { "Content-Type": type, "Content-Length": stat.size, "Accept-Ranges": "bytes" });
    fs.createReadStream(filePath).pipe(res);
  })
  .listen(port, "127.0.0.1", () => {
    console.log(`[serve] catalog  ${ROOT}`);
    console.log(`[serve] weights  ${weights}${fs.existsSync(weights) ? "" : "   (MISSING)"}`);
    console.log(`[serve] http://localhost:${port}/demo/index.html`);
  });
