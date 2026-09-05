// Write target.json for THIS machine: the configuration an entry built here
// would be keyed by.
//
//   node scripts/probe-target.mjs [--out target.json] [--device gpu]
//                                 [--no-perf] [--profile-dir .chrome-profile-probe]
//
// Two kinds of field, and the difference is the point of the file:
//
//   observable: "browser"   a product can read this at runtime, in the page,
//                           with no help. Backend fingerprint, WebGPU adapter
//                           info, userAgentData.
//   observable: "host"      it took a shell on the machine. Chip name, GPU core
//                           count, RAM, the real OS build. A product cannot
//                           match on these; they are here so a human can tell
//                           two configurations apart.
//
// Everything is recorded, nothing is judged. This script does not decide which
// entry suits the machine, and neither does anything else in this repo.
//
// IMPORTANT: chromium.launchPersistentContext(), never chromium.launch().
// Chromium gates the Core ML backend on a non-incognito profile; off the record
// WebNN silently falls back to TFLite/XNNPACK on the CPU and the fingerprint
// this script writes would be a fingerprint of the fallback.

import { chromium } from "playwright";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

const WEBNN_FEATURES = [
  "WebMachineLearningNeuralNetwork",
  "WebMachineLearningNeuralNetworkExperimentalFeatures",
  "WebNNCoreML",
  "WebGPUExperimentalFeatures",
];
const CHROME_ARGS = [
  "--enable-unsafe-webgpu",
  `--enable-features=${WEBNN_FEATURES.join(",")}`,
  "--enable-dawn-features=allow_unsafe_apis",
];

const args = { out: path.join(ROOT, "target.json"), device: "gpu", perf: true, profileDir: path.join(ROOT, ".chrome-profile-probe"), origin: "http://localhost:4173" };
for (let i = 0; i < process.argv.length - 2; i++) {
  const a = process.argv[i + 2];
  const next = () => process.argv[++i + 2];
  if (a === "--out") args.out = path.resolve(next());
  else if (a === "--device") args.device = next();
  else if (a === "--no-perf") args.perf = false;
  else if (a === "--profile-dir") args.profileDir = path.resolve(next());
  else if (a === "-h" || a === "--help") {
    console.log(fs.readFileSync(new URL(import.meta.url), "utf8").split("\n").slice(0, 27).join("\n"));
    process.exit(0);
  } else throw new Error(`unknown argument ${a}`);
}

// ---------------------------------------------------------------------------
// Host-observed: a shell on the machine, never a page.
// ---------------------------------------------------------------------------
const sh = (cmd, argv) => {
  try { return execFileSync(cmd, argv, { encoding: "utf8" }).trim(); } catch { return null; }
};

function hostFacts() {
  const platform = process.platform;
  const base = {
    platform,
    chip: null,
    gpuCores: null,
    memoryGB: null,
    cpuThreads: os.cpus().length || null,
    loadAvg: os.loadavg().map((n) => +n.toFixed(2)),
    unavailable: [],
    observable: "host",
  };
  if (platform === "darwin") {
    base.chip = sh("sysctl", ["-n", "machdep.cpu.brand_string"]);
    const mem = sh("sysctl", ["-n", "hw.memsize"]);
    base.memoryGB = mem ? +(Number(mem) / 2 ** 30).toFixed(0) : null;
    const gpuJson = sh("system_profiler", ["SPDisplaysDataType", "-json"]);
    if (gpuJson) {
      try {
        const g = JSON.parse(gpuJson).SPDisplaysDataType?.[0];
        base.gpuCores = g?.sppci_cores ? parseInt(g.sppci_cores, 10) : null;
      } catch { /* leave null */ }
    }
    const os_ = {
      name: sh("sw_vers", ["-productName"]) ?? "macOS",
      version: sh("sw_vers", ["-productVersion"]),
      build: sh("sw_vers", ["-buildVersion"]),
    };
    return { host: base, os: os_ };
  }
  // Windows and Linux: the shape is here, the readings are not. An entry
  // produced on those platforms records what the browser can see and says so.
  base.unavailable = platform === "win32"
    ? ["chip (wmic/CIM cpu name)", "gpuCores (dxdiag)", "memoryGB (CIM ComputerSystem)"]
    : ["chip (/proc/cpuinfo)", "gpuCores (lspci/vendor tooling)", "memoryGB (/proc/meminfo)"];
  return {
    host: base,
    os: { name: platform === "win32" ? "Windows" : "Linux", version: os.release(), build: null },
  };
}

// ---------------------------------------------------------------------------
// Browser-observable, plus the optional throughput probe.
// ---------------------------------------------------------------------------
async function inPage([deviceType, wantPerf]) {
  const out = { ua: {}, webgpu: null, webnn: {}, perf: null };

  // --- identity ---
  const uad = navigator.userAgentData;
  out.ua.userAgent = navigator.userAgent;
  if (uad) {
    const hi = await uad.getHighEntropyValues([
      "platform", "platformVersion", "fullVersionList", "architecture", "bitness", "model",
    ]);
    out.ua.brands = uad.brands;
    out.ua.high = hi;
  }

  // --- WebGPU adapter ---
  if (!("gpu" in navigator)) out.webgpu = { error: "navigator.gpu missing" };
  else {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) out.webgpu = { error: "requestAdapter returned null" };
    else {
      const ai = adapter.info ?? {};
      const L = adapter.limits;
      out.webgpu = {
        vendor: ai.vendor ?? "",
        architecture: ai.architecture ?? "",
        device: ai.device ?? "",
        description: ai.description ?? "",
        subgroupMinSize: ai.subgroupMinSize ?? null,
        subgroupMaxSize: ai.subgroupMaxSize ?? null,
        features: [...adapter.features].sort(),
        limits: {
          maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
          maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
          maxBufferSize: L.maxBufferSize,
          maxBindGroups: L.maxBindGroups,
          maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
        },
      };
    }
  }

  // --- WebNN backend fingerprint, for every deviceType the browser will give ---
  if (!("ml" in navigator)) out.webnn = { error: "navigator.ml missing (WebNN flag not active?)" };
  else {
    for (const dt of ["gpu", "npu", "cpu"]) {
      try {
        const ctx = await navigator.ml.createContext({ deviceType: dt });
        const lim = ctx.opSupportLimits ? ctx.opSupportLimits() : null;
        const layout = lim?.preferredInputLayout ?? null;
        const rankMax = lim?.input?.rankRange?.max ?? null;
        // Core ML: nchw + input rank max 5. TFLite/XNNPACK: nhwc (+ rank 8).
        // DirectML on Windows: nchw with a higher rank ceiling.
        const name =
          layout === "nchw" && rankMax === 5 ? "coreml"
          : layout === "nhwc" ? "tflite"
          : layout === "nchw" ? "dml"
          : "unknown";
        out.webnn[dt] = {
          ok: true,
          name,
          preferredInputLayout: layout,
          inputRankMax: rankMax,
          inputDataTypes: lim?.input?.dataTypes ?? null,
        };
      } catch (e) {
        out.webnn[dt] = { ok: false, error: String(e) };
      }
    }
  }

  // --- optional: a measured f16 throughput class -------------------------
  //
  // A tiled f16 matmul, 128x128 output tile per workgroup, 8x8 per invocation,
  // f32 accumulate, shapes baked in so the hot loop has no bounds checks. It is
  // a LOWER BOUND on the machine's f16 throughput, not the peak, and it exists
  // only so two entries on different hardware can be put in rough classes.
  if (wantPerf && out.webgpu && !out.webgpu.error) {
    out.perf = await (async () => {
      const M = 2048, N = 2048, K = 2048, BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;
      const budgetMs = 200;
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter?.features.has("shader-f16")) return { error: "shader-f16 not supported" };
      if (typeof Float16Array === "undefined") return { error: "no Float16Array to fill the inputs with" };
      const device = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });

      // Every index into workgroup memory is a literal or a loop counter, and
      // every vector component is a literal swizzle: a dynamic component index
      // scalarises the load and costs about 3x on this class of GPU.
      const acc = [], inner = [], store = [];
      const comp = ["x", "y", "z", "w"];
      for (let m = 0; m < TM; m++) for (let n = 0; n < TN; n++) acc.push(`  var c_${m}_${n} : f32 = 0.0;`);
      for (let m = 0; m < TM; m++)
        inner.push(`      let av${m} = As[(tRow * ${TM}u + ${m}u) * ${BK / 4}u + kq];`);
      for (let kk = 0; kk < 4; kk++) {
        inner.push(`      let bv${kk}_0 = Bs[(kq * 4u + ${kk}u) * ${BN / 4}u + tCol * ${TN / 4}u];`);
        inner.push(`      let bv${kk}_1 = Bs[(kq * 4u + ${kk}u) * ${BN / 4}u + tCol * ${TN / 4}u + 1u];`);
        for (let m = 0; m < TM; m++) for (let n = 0; n < TN; n++)
          inner.push(`      c_${m}_${n} = c_${m}_${n} + f32(av${m}.${comp[kk]}) * f32(bv${kk}_${n >> 2}.${comp[n & 3]});`);
      }
      for (let m = 0; m < TM; m++) for (let n = 0; n < TN; n++)
        store.push(`  C[(rowBase + tRow * ${TM}u + ${m}u) * ${N}u + colBase + tCol * ${TN}u + ${n}u] = c_${m}_${n};`);

      const wgsl = `enable f16;
@group(0) @binding(0) var<storage, read>       A : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read>       B : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
var<workgroup> As : array<vec4<f16>, ${(BM * BK) / 4}>;
var<workgroup> Bs : array<vec4<f16>, ${(BK * BN) / 4}>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  let tRow = lid / 16u;
  let tCol = lid % 16u;
  let rowBase = wg.y * ${BM}u;
  let colBase = wg.x * ${BN}u;
${acc.join("\n")}
  for (var kt : u32 = 0u; kt < ${K}u; kt = kt + ${BK}u) {
    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
      let idx = lid + i * 256u;
      let r = idx / ${BK / 4}u;
      let c = idx % ${BK / 4}u;
      As[idx] = A[((rowBase + r) * ${K}u + kt) / 4u + c];
    }
    for (var i : u32 = 0u; i < 4u; i = i + 1u) {
      let idx = lid + i * 256u;
      let r = idx / ${BN / 4}u;
      let c = idx % ${BN / 4}u;
      Bs[idx] = B[((kt + r) * ${N}u + colBase) / 4u + c];
    }
    workgroupBarrier();
    for (var kq : u32 = 0u; kq < ${BK / 4}u; kq = kq + 1u) {
${inner.join("\n")}
    }
    workgroupBarrier();
  }
${store.join("\n")}
}
`;
      const module = device.createShaderModule({ code: wgsl });
      const info = await module.getCompilationInfo?.();
      const errs = (info?.messages ?? []).filter((m) => m.type === "error");
      if (errs.length) return { error: `shader: ${errs.map((e) => e.message).join("; ")}` };
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });

      const mk = (bytes, usage) => device.createBuffer({ size: bytes, usage, mappedAtCreation: false });
      const A = mk(M * K * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const B = mk(K * N * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const C = mk(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const read = mk(256, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(A, 0, new Float16Array(M * K).fill(0.01));
      device.queue.writeBuffer(B, 0, new Float16Array(K * N).fill(0.02));

      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: A } },
          { binding: 1, resource: { buffer: B } },
          { binding: 2, resource: { buffer: C } },
        ],
      });
      const submit = (n) => {
        const enc = device.createCommandEncoder();
        for (let i = 0; i < n; i++) {
          const p = enc.beginComputePass();
          p.setPipeline(pipeline);
          p.setBindGroup(0, bind);
          p.dispatchWorkgroups(N / BN, M / BM, 1);
          p.end();
        }
        device.queue.submit([enc.finish()]);
      };

      submit(2);
      await device.queue.onSubmittedWorkDone();

      // Check the arithmetic before believing the throughput: every element of
      // C must be K * 0.01 * 0.02, in the f16 rounding of those two literals.
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(C, 0, read, 0, 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0))[0];
      read.unmap();
      const want = K * Number(new Float16Array([0.01])[0]) * Number(new Float16Array([0.02])[0]);
      const checked = Math.abs(got - want) / want < 1e-2;

      let t0 = performance.now();
      submit(4);
      await device.queue.onSubmittedWorkDone();
      const perIter = (performance.now() - t0) / 4;
      const iterations = Math.max(4, Math.min(4000, Math.round(budgetMs / perIter)));

      t0 = performance.now();
      for (let done = 0; done < iterations; done += 20) submit(Math.min(20, iterations - done));
      await device.queue.onSubmittedWorkDone();
      const elapsedMs = performance.now() - t0;

      A.destroy(); B.destroy(); C.destroy(); read.destroy();
      const flops = 2 * M * N * K * iterations;
      return {
        f16TflopsMeasured: +(flops / (elapsedMs / 1000) / 1e12).toFixed(2),
        method: `WGSL tiled f16 matmul ${M}x${N}x${K}, ${BM}x${BN} workgroup tile, ${TM}x${TN} per invocation, f32 accumulate, ${iterations} dispatches over ~${budgetMs} ms. A LOWER BOUND on this machine's f16 throughput, not its peak.`,
        shape: [M, N, K],
        iterations,
        elapsedMs: +elapsedMs.toFixed(1),
        checked,
        error: checked ? null : `kernel output ${got} != expected ${want}; throughput not believed`,
      };
    })().catch((e) => ({ error: String(e) }));
  }

  return out;
}

// ---------------------------------------------------------------------------
const { host, os: hostOs } = hostFacts();

fs.mkdirSync(args.profileDir, { recursive: true });
const context = await chromium.launchPersistentContext(args.profileDir, {
  channel: "chrome",
  headless: false,
  args: CHROME_ARGS,
  viewport: { width: 900, height: 600 },
});
const page = await context.newPage();
page.on("pageerror", (e) => console.error("[page]", e.message));
// WebGPU and WebNN are secure-context only and about:blank is not one. A route
// interception is enough of a localhost origin; no server needed.
await page.route(`${args.origin}/**`, (route) =>
  route.fulfill({ contentType: "text/html", body: "<!doctype html><title>webnn-catalog probe</title>" }));
await page.goto(`${args.origin}/probe`);

console.log(`[probe] chrome ${context.browser()?.version() ?? "?"} (persistent profile ${path.relative(ROOT, args.profileDir)})`);
const seen = await page.evaluate(inPage, [args.device, args.perf]);
await context.close();

// ---------------------------------------------------------------------------
// fullVersionList carries the product brand ("Google Chrome"), the engine
// ("Chromium") and a deliberate decoy ("Not)A;Brand"). The product is the one
// an entry is keyed by; the engine goes in `engine` because two products can
// share it.
const list = seen.ua?.high?.fullVersionList ?? seen.ua?.brands ?? [];
const real = list.filter((b) => !/Not.?A.?Brand/i.test(b.brand));
const brand = real.find((b) => b.brand !== "Chromium") ?? real[0] ?? { brand: "unknown", version: "0" };
brand.brand = brand.brand.replace(/^Google /, "").replace(/^Microsoft /, "");
const chromium_ = list.find((b) => b.brand === "Chromium");
const fp = seen.webnn?.[args.device] ?? {};

const target = {
  generatedAt: new Date().toISOString(),
  tool: "webnn-catalog scripts/probe-target.mjs",
  backend: {
    name: fp.name ?? "unknown",
    deviceType: args.device,
    fingerprint: {
      preferredInputLayout: fp.preferredInputLayout ?? null,
      inputRankMax: fp.inputRankMax ?? null,
      inputDataTypes: fp.inputDataTypes ?? null,
    },
    byDeviceType: Object.fromEntries(Object.entries(seen.webnn ?? {}).filter(([k]) => k !== "error")),
    observable: "browser",
  },
  browser: {
    name: brand.brand,
    version: brand.version,
    major: parseInt(String(brand.version).split(".")[0], 10),
    ...(chromium_ ? { engine: `Chromium ${chromium_.version}` } : {}),
    userAgent: seen.ua?.userAgent ?? "",
    flags: CHROME_ARGS,
    profile: "persistent",
    observable: "browser",
  },
  os: {
    name: hostOs.name ?? "unknown",
    version: hostOs.version ?? seen.ua?.high?.platformVersion ?? "unknown",
    major: parseInt(String(hostOs.version ?? "0").split(".")[0], 10),
    ...(hostOs.build ? { build: hostOs.build } : {}),
    browserReported: {
      platform: seen.ua?.high?.platform ?? null,
      platformVersion: seen.ua?.high?.platformVersion ?? null,
      architecture: seen.ua?.high?.architecture ?? null,
      bitness: seen.ua?.high?.bitness ?? null,
      note: "What a page can see on its own. `version` and `build` above are the host reading, which is authoritative where the two disagree: userAgentData's platformVersion is capped or coarsened on some platforms and browser builds.",
    },
    observable: "browser(partial)/host",
  },
  gpu: {
    vendor: seen.webgpu?.vendor ?? "",
    architecture: seen.webgpu?.architecture ?? "",
    device: seen.webgpu?.device ?? "",
    description: seen.webgpu?.description ?? "",
    subgroupMinSize: seen.webgpu?.subgroupMinSize ?? null,
    subgroupMaxSize: seen.webgpu?.subgroupMaxSize ?? null,
    features: seen.webgpu?.features ?? [],
    limits: seen.webgpu?.limits ?? {},
    observable: "browser",
  },
  host,
  ...(seen.perf ? { perfClass: { ...seen.perf, f16TflopsMeasured: seen.perf.f16TflopsMeasured ?? null, observable: "browser(probe)" } } : {}),
};

fs.writeFileSync(args.out, JSON.stringify(target, null, 2) + "\n");
console.log(JSON.stringify(target, null, 2));
console.log(`\nwrote ${path.relative(process.cwd(), args.out)}`);
if (target.backend.name !== "coreml" && process.platform === "darwin")
  console.log(`\nNOTE: backend is "${target.backend.name}", not "coreml". On macOS that means the CPU fallback: check the profile is not off the record.`);
