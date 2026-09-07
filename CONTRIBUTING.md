# Contributing

The [README](README.md) is for consuming the catalog. This file is for adding to
it: a timing row from your machine, an entry for new hardware, or a new family.

Entries come in two kinds, and `runtimeKind` discriminates them. A **WebNN**
entry carries a recipe — the exact `MLGraphBuilder` call sequence — plus a
constants blob. A **WebLLM** entry names hash-pinned runtime JavaScript and
model-library WASM plus a revision-pinned upstream model repository.

## 1. A measurement row

The cheapest and most useful contribution. Every entry has been measured on one
machine, which makes its timings a single data point. Running an existing entry
on different hardware turns it into a range.

A second machine running the same entry is a **row**, not an entry:

```bash
node scripts/probe-target.mjs --out target.json
node scripts/add-measurement.mjs --entry <family>/<entry-id> \
  --results <results>.json --target target.json --protocol e2e
```

Each row carries its own host fingerprint, so one entry accumulates timings from
several machines without becoming several entries.

**Rules that are not negotiable.** A number without a machine, an OS, a browser
build and a protocol is not a measurement. A row's load average is never borrowed
from another moment. A projection is labelled as one and never presented as
measured. And do not record what else was running on your machine by name —
describe load generically or not at all.

## 2. An entry

### WebNN

You need a tuning run that recorded its `MLGraphBuilder` call sequence, plus the
constants blob that sequence indexes.

```bash
# 1. Describe the machine you are about to build on.
node scripts/probe-target.mjs --out target.json

# 2. Write the entry from what the tuning run produced.
node scripts/add-entry.mjs \
  --family sd-turbo-512-1step \
  --target target.json \
  --recipe image=<dump>.json --recipe text=<dump>.json \
  --constants image=<blob>.bin --constants text=<blob>.bin \
  --chain text.out=image.encoder_hidden_states \
  --variant exact \
  --verification <dir> --provenance <ledger>.md \
  --measurements <results>.json \
  --produced-by "webnn-workbench@<commit>"

# 3. Check it, against the schemas and against the hardware.
node scripts/validate.mjs
node scripts/verify.mjs --entry sd-turbo-512-1step/<entry-id>
```

`add-entry.mjs` reads the recipes to fill in the entry's I/O, hashes the
constants blobs **where they lie** (they are gigabytes and never enter git),
writes `entry.json`, `manifest.json` and `measurements.json`, copies the
verification set and the ledger in, adds a summary row to `catalog.json`, and
refuses to overwrite without `--force`.

### WebLLM

A WebLLM entry publishes hash-pinned artifacts and points at immutable upstream
weights. It never rehosts the model.

```bash
node scripts/verify-artifacts.mjs --entry <family>/<entry-id>
WEBNN_MAX_LOAD=0 npm run verify:webllm
```

`verify-artifacts.mjs` checks every declared byte count and SHA-256.
`verify:webllm` runs the family's greedy corpus and requires byte-identical
output against the baseline the entry names. `scripts/gpu-lock.mjs` serializes
runs so two benchmarks never share a GPU.

Each WebLLM entry ships a `NOTICE.md` with third-party attribution, and records
`source.license` with an SPDX identifier pinned to the upstream revision. Do not
add one without both.

### Every entry needs a ledger

`provenance.md` is not optional and not a formality. It records what was folded
and what it was worth; what was tried and rejected, with the number that killed
it; and what the backend turned out to be like.

The op list survives on its own. The reasoning does not.

### Publishing constants

```bash
node scripts/chunk-manifest.mjs --entry <family>/<entry-id>
scripts/publish-weights.sh <family>/<entry-id>
```

`chunk-manifest.mjs` tiles a blob into Range-sized chunks at a 64 MiB target,
putting every boundary on a constant start — and for a blob several recipes read,
on a constant start in all of them — so nothing straddles a seam and every view
stays zero-copy. The host must answer Range requests and be CORS-open.

## 3. A family

`family.json` is authored by hand. A tool that guessed at a contract would be
guessing at the one thing entries are not allowed to disagree about.

A family is one model, one task, one I/O contract. Every entry meets it exactly.
**If a change would falsify any of it, that is a new family, not a new entry** —
different resolution, step count, tokenizer or output shape all qualify.

## Checks

```bash
npm install
node scripts/validate.mjs          # schema + cross-file checks
node scripts/check-chunks.mjs      # replay graphs off the chunk lists
npm test                           # loader unit tests
node scripts/verify.mjs --entry <family>/<entry-id> --weights <dir>
node scripts/serve.mjs             # then open demo/index.html or demo/qwen.html
```

`validate.mjs` is dependency-free. It implements the subset of JSON Schema the
schemas use, refuses a keyword it does not know rather than passing it silently,
and then checks what a schema cannot: that referenced files exist, that each
recipe still hashes to what its manifest says, that an entry's declared I/O
equals its recipes' and meets its family's contract, that every chain link
typechecks, and that the index agrees with the entries it indexes.

## Known gaps

**The ToDo fast variant of SD-Turbo has no entry.** Token Downsampling runs at
54.6 ms, about 20% faster, for a visibly different image (PSNR 24.3 dB against
the exact pipeline). It was measured end to end but its `MLGraphBuilder` call
sequence was never recorded, so there is no recipe. The numbers are in the
SD-Turbo entry's
[`provenance.md`](families/sd-turbo-512-1step/entries/coreml-apple-m5-pro-macos26-chrome152/provenance.md#the-optional-fast-variant).

**Every entry targets Apple silicon.** A DirectML or TFLite WebNN entry, or a
WebLLM entry on a non-Apple GPU, would be the highest-value contribution here —
it is the case the family/entry split exists to serve, and nothing exercises it.

**Recipe schema v1 cannot distinguish call forms.** The recorder flattens
positional non-operand arguments into the options bag, so the JSON alone cannot
tell `softmax(x, 2)` from `softmax(x, {axis: 2})`; `x-callForms.positional` makes
it recoverable. A v2 should record the argument list positionally.
