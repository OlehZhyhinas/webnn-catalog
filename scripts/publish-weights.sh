#!/usr/bin/env bash
# Publish a catalog entry's constants blobs to a Hugging Face repo and write the
# resulting URLs back into its manifest.json.
#
#   scripts/publish-weights.sh --repo <user>/<name> [options]
#
# THIS SCRIPT IS NOT RUN AUTOMATICALLY. It uploads gigabytes to a public-by-
# default host. Read it, set --repo, and run it yourself.
#
# Options:
#   --repo <id>       target repo, e.g. olehzhyhinas/webnn-catalog-sd-turbo   (required)
#   --type <t>        dataset | model                      (default: dataset)
#   --entry <ref>     catalog entry as <family>/<entry-id>
#                     (default: sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152)
#   --weights <dir>   directory holding the .bin files
#                     (default: ../webnn-workbench/bench/webnn/ir)
#   --revision <rev>  branch or tag to upload to           (default: main)
#   --private         create the repo private
#   --dry-run         print the commands, upload nothing
#   --skip-upload     only rewrite manifest.json from an already-published repo
#
# Weights are the published artifact; the recipes are the repo. A .bin is the
# exact byte stream the recipe's `constants` offsets index into, so a mismatch
# is not a degradation, it is a wrong graph. Every step below verifies sha256
# against the manifest before and after.

set -euo pipefail

REPO=""
REPO_TYPE="dataset"
ENTRY="sd-turbo-512-1step/coreml-apple-m5-pro-macos26-chrome152"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="$(cd "$ROOT/.." && pwd)/webnn-workbench/bench/webnn/ir"
REVISION="main"
PRIVATE=""
DRY_RUN=0
SKIP_UPLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --type) REPO_TYPE="$2"; shift 2 ;;
    --entry) ENTRY="$2"; shift 2 ;;
    --weights) WEIGHTS="$(cd "$2" && pwd)"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --private) PRIVATE="--private"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-upload) SKIP_UPLOAD=1; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$REPO" ]] || { echo "error: --repo is required (e.g. --repo someone/webnn-catalog-sd-turbo)" >&2; exit 2; }

FAMILY="${ENTRY%%/*}"
ENTRY_ID="${ENTRY##*/}"
MANIFEST="$ROOT/families/$FAMILY/entries/$ENTRY_ID/manifest.json"
[[ -f "$MANIFEST" ]] || { echo "error: no manifest at $MANIFEST" >&2; exit 2; }

run() {
  if [[ $DRY_RUN -eq 1 ]]; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi
}

command -v huggingface-cli >/dev/null || {
  echo "error: huggingface-cli not found. pip install -U 'huggingface_hub[cli]' and huggingface-cli login" >&2
  exit 2
}

echo "entry     $ENTRY"
echo "manifest  $MANIFEST"
echo "weights   $WEIGHTS"
echo "target    $REPO_TYPE $REPO @ $REVISION"
echo

# ---------------------------------------------------------------------------
# 1. Check the local blobs against the manifest BEFORE uploading anything.
# ---------------------------------------------------------------------------
FILES=$(node -e '
  const m = require(process.argv[1]);
  for (const [k, c] of Object.entries(m.constants)) console.log([k, c.file, c.bytes, c.sha256].join("\t"));
' "$MANIFEST")

echo "verifying local blobs"
while IFS=$'\t' read -r key file bytes sha; do
  path="$WEIGHTS/$file"
  [[ -f "$path" ]] || { echo "  error: missing $path" >&2; exit 2; }
  actual_bytes=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path")
  [[ "$actual_bytes" == "$bytes" ]] || { echo "  error: $file is $actual_bytes bytes, manifest says $bytes" >&2; exit 2; }
  echo "  hashing $file ($((bytes / 1024 / 1024)) MiB)..."
  actual_sha=$(shasum -a 256 "$path" | cut -d' ' -f1)
  [[ "$actual_sha" == "$sha" ]] || { echo "  error: $file sha256 $actual_sha != manifest $sha" >&2; exit 2; }
  echo "  ok $key  $file  $bytes bytes  ${sha:0:16}..."
done <<< "$FILES"
echo

# ---------------------------------------------------------------------------
# 2. Create the repo (idempotent) and upload.
# ---------------------------------------------------------------------------
if [[ $SKIP_UPLOAD -eq 0 ]]; then
  echo "creating $REPO_TYPE repo $REPO (no-op if it exists)"
  run huggingface-cli repo create "$REPO" --type "$REPO_TYPE" $PRIVATE -y || true
  echo

  echo "uploading"
  while IFS=$'\t' read -r key file bytes sha; do
    echo "  $file -> $REPO:$ENTRY/$file"
    # huggingface-cli upload <repo> <local> <path-in-repo>. Uploads are
    # resumable and chunked; a 1.65 GB blob is well within LFS limits but will
    # take a while on a domestic uplink.
    run huggingface-cli upload "$REPO" "$WEIGHTS/$file" "$ENTRY/$file" \
      --repo-type "$REPO_TYPE" --revision "$REVISION" \
      --commit-message "$ENTRY: $file ($bytes bytes, sha256 $sha)"
  done <<< "$FILES"
  echo
fi

# ---------------------------------------------------------------------------
# 3. Write the resulting URLs into manifest.json.
# ---------------------------------------------------------------------------
# resolve/<revision>/ is the raw-bytes endpoint and answers Range requests,
# which is what the loader's chunked constant path needs.
BASE="https://huggingface.co/$( [[ "$REPO_TYPE" == dataset ]] && echo "datasets/" )$REPO/resolve/$REVISION/$ENTRY"

echo "writing URLs into $MANIFEST"
echo "  base $BASE"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "  [dry-run] manifest unchanged"
else
  node -e '
    const fs = require("fs");
    const [manifestPath, base] = process.argv.slice(1);
    const m = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
    for (const [k, c] of Object.entries(m.constants)) {
      c.url = `${base}/${c.file}`;
      console.log(`  ${k} -> ${c.url}`);
    }
    m.published = { at: new Date().toISOString(), base };
    fs.writeFileSync(manifestPath, JSON.stringify(m, null, 2) + "\n");
  ' "$MANIFEST" "$BASE"
fi
echo

# ---------------------------------------------------------------------------
# 4. Verify what is actually being served: sha256 the published bytes back.
# ---------------------------------------------------------------------------
if [[ $DRY_RUN -eq 0 && $SKIP_UPLOAD -eq 0 ]]; then
  echo "verifying published bytes (streams each blob back; skip with --skip-upload next time)"
  while IFS=$'\t' read -r key file bytes sha; do
    url="$BASE/$file"
    echo "  GET $url"
    remote_sha=$(curl -sSL "$url" | shasum -a 256 | cut -d' ' -f1)
    if [[ "$remote_sha" == "$sha" ]]; then echo "  ok $file"; else echo "  MISMATCH $file: $remote_sha != $sha" >&2; exit 1; fi
  done <<< "$FILES"
  echo
fi

cat <<EOF
done.

next:
  1. commit the manifest change
  2. node scripts/verify.mjs --weights <local dir>   (still passes against local bytes)
  3. serve the catalog with no --weights mount and confirm the page loads from
     the published URLs, since manifest.constants[*].url now wins over baseUrl
EOF
