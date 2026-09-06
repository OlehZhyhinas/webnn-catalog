import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import test from "node:test";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const {
  fetchVerifiedArtifact,
  isGreedyBurstEligible,
  sha256Hex,
} = await import("../runtime/webllm-loader.js");

test("greedy burst eligibility matches runtime guardrails", () => {
  assert.equal(isGreedyBurstEligible({ temperature: 0 }), true);
  assert.equal(isGreedyBurstEligible({ temperature: 0, logprobs: true }), false);
  assert.equal(
    isGreedyBurstEligible({ temperature: 0, logit_bias: { 42: 1 } }),
    false,
  );
  assert.equal(
    isGreedyBurstEligible({ temperature: 0, repetition_penalty: 1.01 }),
    false,
  );
  assert.equal(
    isGreedyBurstEligible({ temperature: 0, response_format: { type: "json_object" } }),
    false,
  );
});

test("artifact bytes are checked before use", async () => {
  const bytes = new TextEncoder().encode("verified runtime");
  const artifact = {
    id: "runtime",
    kind: "javascript-module",
    url: "https://example.test/runtime.js",
    bytes: bytes.byteLength,
    sha256: await sha256Hex(bytes),
  };
  const fetchImpl = async () => new Response(bytes);
  const loaded = await fetchVerifiedArtifact(artifact, { fetchImpl });
  assert.equal(loaded.sha256, artifact.sha256);

  await assert.rejects(
    fetchVerifiedArtifact({ ...artifact, bytes: bytes.byteLength + 1 }, { fetchImpl }),
    /expected .* bytes/,
  );
  await assert.rejects(
    fetchVerifiedArtifact({ ...artifact, sha256: "0".repeat(64) }, { fetchImpl }),
    /does not match/,
  );
});

test("non-HTTPS artifacts are rejected", async () => {
  await assert.rejects(
    fetchVerifiedArtifact({
      id: "runtime",
      kind: "javascript-module",
      url: "http://example.test/runtime.js",
      bytes: 1,
      sha256: "0".repeat(64),
    }),
    /must use HTTPS/,
  );
});
