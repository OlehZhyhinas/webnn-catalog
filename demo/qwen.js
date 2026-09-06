import { loadWebLLMFromUrl } from "../runtime/webllm-loader.js";

const ENTRY =
  new URL(
    "../families/qwen3-0.6b-q4f16-1/entries/" +
      "webgpu-apple-m5-pro-macos26-chrome152-sg32-burst4/entry.json",
    location.href,
  ).href;

const loadButton = document.querySelector("#load");
const runButton = document.querySelector("#run");
const prompt = document.querySelector("#prompt");
const status = document.querySelector("#status");
const output = document.querySelector("#output");
let rig;

loadButton.addEventListener("click", async () => {
  loadButton.disabled = true;
  status.textContent = "Checking capabilities and downloading verified artifacts…";
  try {
    rig = await loadWebLLMFromUrl(ENTRY, {
      allowLocalhost: location.hostname === "localhost",
      onProgress: (progress) => {
        status.textContent =
          progress?.text ?? `${progress?.phase ?? "loading"}…`;
      },
    });
    status.textContent = "Ready. Runtime JS and model WASM hashes verified.";
    runButton.disabled = false;
  } catch (error) {
    status.textContent = `Load failed: ${error.message}`;
    loadButton.disabled = false;
  }
});

runButton.addEventListener("click", async () => {
  runButton.disabled = true;
  output.textContent = "";
  status.textContent = "Generating…";
  try {
    await rig.engine.resetChat(false);
    const chunks = await rig.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt.value }],
      temperature: 0,
      max_tokens: 128,
      stream: true,
      extra_body: { enable_thinking: false },
    });
    for await (const chunk of chunks) {
      output.textContent += chunk.choices[0]?.delta?.content ?? "";
    }
    status.textContent = "Done.";
  } catch (error) {
    status.textContent = `Generation failed: ${error.message}`;
  } finally {
    runButton.disabled = false;
  }
});

addEventListener("pagehide", () => {
  void rig?.dispose();
});
