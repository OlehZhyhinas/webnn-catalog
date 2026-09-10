import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";
import test from "node:test";

if (!globalThis.crypto) globalThis.crypto = webcrypto;

const {
  fetchVerifiedArtifact,
  isGreedyBurstEligible,
  parsePromptLookup,
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

test("promptLookup parser accepts full spec and defaults", () => {
  assert.deepEqual(parsePromptLookup("5:3:2:fork"), {
    k: 5,
    nMax: 3,
    nMin: 2,
    hybrid: "fork",
  });
  assert.deepEqual(parsePromptLookup("5"), {
    k: 5,
    nMax: 3,
    nMin: 2,
  });
  assert.deepEqual(parsePromptLookup("5:4"), {
    k: 5,
    nMax: 4,
    nMin: 2,
  });
});

test("promptLookup parser clamps nMin to nMax", () => {
  assert.deepEqual(parsePromptLookup("5:2:7"), {
    k: 5,
    nMax: 2,
    nMin: 2,
  });
  assert.deepEqual(parsePromptLookup("5:fork"), {
    k: 5,
    nMax: 3,
    nMin: 2,
    hybrid: "fork",
  });
});

test("promptLookup parser rejects malformed specs", () => {
  for (const value of [
    null,
    undefined,
    "",
    "0",
    "5:0",
    "5::2",
    "5:3:2:fork:extra",
    "5:3:2:Fork",
    " 5:3:2 ",
  ]) {
    assert.equal(parsePromptLookup(value), null);
  }
});
