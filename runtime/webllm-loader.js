// Verified loader for catalog entries whose runtimeKind is "webllm".
//
// Runtime JavaScript and model-library WASM are fetched as bytes, checked
// against the entry's artifact manifest, and only then exposed through blob
// URLs. Model weights remain in the revision-pinned upstream MLC repository.

const HEX_256 = /^[0-9a-f]{64}$/;
const PROMPT_LOOKUP_SPEC = /^[1-9]\d*(?::[1-9]\d*){0,2}(?::fork)?$/;

const joinUrl = (base, value) =>
  new URL(value, String(base).endsWith("/") ? base : `${base}/`).href;

function assertPublicHttps(url, { allowLocalhost = false } = {}) {
  const parsed = new URL(url);
  const local =
    allowLocalhost &&
    (parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1");
  if (parsed.protocol !== "https:" && !local) {
    throw new Error(`catalog artifacts must use HTTPS: ${url}`);
  }
  if (parsed.username || parsed.password) {
    throw new Error(`catalog artifact URL must not contain credentials: ${url}`);
  }
  return parsed.href;
}

export async function sha256Hex(bytes) {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

export async function fetchVerifiedArtifact(
  artifact,
  { fetchImpl = fetch, allowLocalhost = false } = {},
) {
  if (!Number.isSafeInteger(artifact.bytes) || artifact.bytes <= 0) {
    throw new Error(`${artifact.name ?? artifact.id ?? "artifact"}: invalid byte count`);
  }
  if (!HEX_256.test(artifact.sha256)) {
    throw new Error(`${artifact.name ?? artifact.id ?? "artifact"}: invalid SHA-256`);
  }
  const url = assertPublicHttps(artifact.url, { allowLocalhost });
  const response = await fetchImpl(url, { mode: "cors", credentials: "omit" });
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== artifact.bytes) {
    throw new Error(
      `${artifact.name ?? artifact.id ?? "artifact"}: expected ${artifact.bytes} bytes, received ${bytes.byteLength}`,
    );
  }
  const actual = await sha256Hex(bytes);
  if (actual !== artifact.sha256) {
    throw new Error(
      `${artifact.name ?? artifact.id ?? "artifact"}: SHA-256 ${actual} does not match ${artifact.sha256}`,
    );
  }
  return { bytes, url, sha256: actual };
}

export async function probeWebLLMCapabilities({
  navigatorImpl = navigator,
  powerPreference = "high-performance",
} = {}) {
  if (!navigatorImpl.gpu) {
    return { ok: false, reason: "WebGPU is unavailable" };
  }
  const adapter = await navigatorImpl.gpu.requestAdapter({ powerPreference });
  if (!adapter) return { ok: false, reason: "WebGPU returned no adapter" };
  const features = [...adapter.features].sort();
  const minSubgroupSize =
    adapter.limits?.minSubgroupSize ?? adapter.info?.subgroupMinSize ?? null;
  const maxSubgroupSize =
    adapter.limits?.maxSubgroupSize ?? adapter.info?.subgroupMaxSize ?? null;
  const subgroup32 =
    features.includes("subgroups") &&
    (minSubgroupSize == null || minSubgroupSize <= 32) &&
    (maxSubgroupSize == null || maxSubgroupSize >= 32);
  return {
    ok: subgroup32,
    reason: subgroup32 ? null : "the tuned path requires WebGPU subgroups with size 32",
    adapter,
    features,
    minSubgroupSize,
    maxSubgroupSize,
    info: adapter.info ?? null,
  };
}

export function isGreedyBurstEligible(request = {}) {
  const temperature = request.temperature ?? 1;
  const frequency = request.frequency_penalty ?? 0;
  const presence = request.presence_penalty ?? 0;
  const repetition = request.repetition_penalty ?? 1;
  const hasBias =
    request.logit_bias != null && Object.keys(request.logit_bias).length > 0;
  const hasGrammar = Boolean(
    request.grammar ||
      request.json_schema ||
      request.response_format ||
      request.hasGrammarMatcher,
  );
  return (
    temperature === 0 &&
    !request.logprobs &&
    !hasBias &&
    frequency === 0 &&
    presence === 0 &&
    repetition === 1 &&
    !hasGrammar &&
    !request.hasLogitProcessor
  );
}

export function parsePromptLookup(spec) {
  if (typeof spec !== "string") return null;
  if (!PROMPT_LOOKUP_SPEC.test(spec)) return null;
  const parts = spec.split(":");
  const hybrid = parts.at(-1) === "fork" ? parts.pop() : null;
  const [kText, nMaxText, nMinText] = parts;
  const k = Number(kText);
  const nMax = Number(nMaxText ?? 3);
  const nMin = Math.min(Number(nMinText ?? 2), nMax);
  return {
    k,
    nMax,
    nMin,
    ...(hybrid === "fork" ? { hybrid: "fork" } : {}),
  };
}

async function fetchJson(url, fetchImpl) {
  const response = await fetchImpl(url, {
    mode: "cors",
    credentials: "omit",
  });
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
  return response.json();
}

/**
 * Load an already-parsed WebLLM catalog entry.
 *
 * Returns an OpenAI-compatible WebLLM engine plus a dispose() method. The
 * runtime itself re-checks every generation request: ineligible requests use
 * the normal one-token sampler even though the entry is configured for K=4.
 */
export async function loadWebLLMEntry(entry, options = {}) {
  if (entry.runtimeKind !== "webllm") {
    throw new Error(`entry ${entry.id} is ${entry.runtimeKind}, not webllm`);
  }
  const {
    baseUrl = location.href,
    fetchImpl = fetch,
    navigatorImpl = navigator,
    onProgress,
    allowLocalhost = false,
  } = options;
  const capability = await probeWebLLMCapabilities({ navigatorImpl });
  if (!capability.ok) throw new Error(capability.reason);

  const manifestUrl = assertPublicHttps(joinUrl(baseUrl, entry.artifacts), {
    allowLocalhost,
  });
  const manifest = await fetchJson(manifestUrl, fetchImpl);
  const runtimeArtifact = manifest.runtimeJs?.[0];
  const wasmArtifact = manifest.modelWasm?.[0];
  const modelArtifact = manifest.modelResources?.find(
    (artifact) => artifact.revision === entry.model.revision,
  );
  if (!runtimeArtifact || !wasmArtifact || !modelArtifact) {
    throw new Error("artifact manifest must include runtime JS, model WASM, and model repository");
  }

  onProgress?.({ phase: "runtime-download", progress: 0 });
  const [runtimeFile, wasmFile] = await Promise.all([
    fetchVerifiedArtifact(runtimeArtifact, { fetchImpl, allowLocalhost }),
    fetchVerifiedArtifact(wasmArtifact, { fetchImpl, allowLocalhost }),
  ]);
  onProgress?.({ phase: "runtime-download", progress: 1 });

  const runtimeUrl = URL.createObjectURL(
    new Blob([runtimeFile.bytes], { type: "text/javascript" }),
  );
  const wasmUrl = URL.createObjectURL(
    new Blob([wasmFile.bytes], { type: "application/wasm" }),
  );

  const previous = {
    burst: globalThis.__webllmGreedyBurst,
    argmax: globalThis.__webllmGreedyArgmax,
    cleanup: globalThis.__webllmDeferDecodeCleanup,
    batchPass: globalThis.__tvmjsWebGPUBatchPass,
    flushEvery: globalThis.__tvmjsWebGPUFlushEvery,
    bindCache: globalThis.__tvmjsWebGPUBindGroupCache,
    lookahead: globalThis.__webllmBurstLookahead,
    promptLookup: globalThis.__webllmPromptLookup,
  };
  globalThis.__webllmGreedyBurst = entry.runtime.config.greedyBurst;
  globalThis.__webllmGreedyArgmax = true;
  globalThis.__webllmDeferDecodeCleanup = false;
  globalThis.__tvmjsWebGPUBatchPass = entry.runtime.config.batchPass === true;
  globalThis.__tvmjsWebGPUFlushEvery =
    Number(entry.runtime.config.flushEvery) > 0
      ? Number(entry.runtime.config.flushEvery)
      : 0;
  globalThis.__tvmjsWebGPUBindGroupCache =
    entry.runtime.config.bindGroupCache === true;
  // Decode steps kept queued on the GPU past the burst being read back.
  globalThis.__webllmBurstLookahead =
    Number(entry.runtime.config.lookahead) > 0
      ? Number(entry.runtime.config.lookahead)
      : 0;
  globalThis.__webllmPromptLookup = parsePromptLookup(
    entry.runtime.config.promptLookup,
  );

  let engine;
  try {
    const webllm = await import(runtimeUrl);
    const modelRecord = {
      model: modelArtifact.url.replace(/\/$/, ""),
      model_id: entry.model.id,
      model_lib: wasmUrl,
      vram_required_MB: entry.model.config.vramRequiredMB,
      low_resource_required: entry.model.config.lowResourceRequired ?? false,
      overrides: entry.model.config.overrides ?? {},
    };
    engine = await webllm.CreateMLCEngine(entry.model.id, {
      appConfig: { model_list: [modelRecord] },
      initProgressCallback: onProgress,
      ...(entry.runtime.config.engineConfig ?? {}),
    });
    return {
      engine,
      entry,
      manifest,
      capability,
      isFastPathEligible: isGreedyBurstEligible,
      async dispose() {
        try {
          await engine?.unload?.();
        } finally {
          URL.revokeObjectURL(runtimeUrl);
          URL.revokeObjectURL(wasmUrl);
          globalThis.__webllmGreedyBurst = previous.burst;
          globalThis.__webllmGreedyArgmax = previous.argmax;
          globalThis.__webllmDeferDecodeCleanup = previous.cleanup;
          globalThis.__tvmjsWebGPUBatchPass = previous.batchPass;
          globalThis.__tvmjsWebGPUFlushEvery = previous.flushEvery;
          globalThis.__tvmjsWebGPUBindGroupCache = previous.bindCache;
          globalThis.__webllmBurstLookahead = previous.lookahead;
          globalThis.__webllmPromptLookup = previous.promptLookup;
        }
      },
    };
  } catch (error) {
    URL.revokeObjectURL(runtimeUrl);
    URL.revokeObjectURL(wasmUrl);
    globalThis.__webllmGreedyBurst = previous.burst;
    globalThis.__webllmGreedyArgmax = previous.argmax;
    globalThis.__webllmDeferDecodeCleanup = previous.cleanup;
    globalThis.__tvmjsWebGPUBatchPass = previous.batchPass;
    globalThis.__tvmjsWebGPUFlushEvery = previous.flushEvery;
    globalThis.__tvmjsWebGPUBindGroupCache = previous.bindCache;
    globalThis.__webllmBurstLookahead = previous.lookahead;
    globalThis.__webllmPromptLookup = previous.promptLookup;
    throw error;
  }
}

export async function loadWebLLMFromUrl(entryUrl, options = {}) {
  const fetchImpl = options.fetchImpl ?? fetch;
  const url = assertPublicHttps(entryUrl, {
    allowLocalhost: options.allowLocalhost,
  });
  const entry = await fetchJson(url, fetchImpl);
  return loadWebLLMEntry(entry, {
    ...options,
    fetchImpl,
    baseUrl: new URL(".", url).href,
  });
}
