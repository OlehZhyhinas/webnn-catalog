// Texo's tokenizer: HuggingFace `tokenizers` WordLevel over WhitespaceSplit,
// 687 entries, TemplateProcessing "<s> A </s>", no normalizer, no decoder
// object. Decoding is a table lookup joined with spaces; LatexGen's recorded
// strings additionally have transformers' clean_up_tokenization applied
// (tokenizer_config.json: clean_up_tokenization_spaces = true), which is why
// they read `\mathbf { r }, t` and not `\mathbf { r } , t`.
//
//   const tok = await TexoTokenizer.load("families/texo-384/tokenizer/");
//   tok.decode([678, 652, ...])               // "x ^ { 2 } + y ^ { 2 } = r ^ { 2 }"
//   tok.decode(ids, { cleanUp: false })       // raw space-joined tokens
//   tok.encode("x ^ { 2 }")                   // [0, 678, 652, ..., 2]

export default class TexoTokenizer {
  constructor(json) {
    if (json.model?.type !== "WordLevel") throw new Error(`expected a WordLevel tokenizer, got ${json.model?.type}`);
    if (json.pre_tokenizer?.type !== "WhitespaceSplit") throw new Error(`expected WhitespaceSplit, got ${json.pre_tokenizer?.type}`);
    this.vocab = json.model.vocab; // token -> id
    this.unk = json.model.unk_token ?? "<unk>";
    this.id2tok = new Map(Object.entries(this.vocab).map(([t, i]) => [i, t]));
    this.special = new Set((json.added_tokens ?? []).filter((t) => t.special).map((t) => t.id));
    const sp = Object.fromEntries((json.post_processor?.special_tokens ? Object.values(json.post_processor.special_tokens) : []).map((s) => [s.id, s.ids?.[0]]));
    this.bos = sp["<s>"] ?? this.vocab["<s>"] ?? 0;
    this.eos = sp["</s>"] ?? this.vocab["</s>"] ?? 2;
    this.pad = this.vocab["<pad>"] ?? 1;
  }

  static async load(baseUrl) {
    const url = `${baseUrl.replace(/\/$/, "")}/tokenizer.json`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return new TexoTokenizer(await r.json());
  }

  /** transformers.PreTrainedTokenizer.clean_up_tokenization, verbatim. */
  static cleanUp(s) {
    return s
      .replaceAll(" .", ".").replaceAll(" ?", "?").replaceAll(" !", "!").replaceAll(" ,", ",")
      .replaceAll(" ' ", "'").replaceAll(" n't", "n't").replaceAll(" 'm", "'m").replaceAll(" 's", "'s")
      .replaceAll(" 've", "'ve").replaceAll(" 're", "'re");
  }

  decode(ids, { skipSpecial = true, cleanUp = true } = {}) {
    const parts = [];
    for (const id of ids) {
      if (skipSpecial && this.special.has(id)) continue;
      const t = this.id2tok.get(id);
      if (t === undefined) throw new Error(`token id ${id} is not in the ${this.id2tok.size}-entry vocabulary`);
      parts.push(t);
    }
    const s = parts.join(" ");
    return cleanUp ? TexoTokenizer.cleanUp(s) : s;
  }

  /** "<s> A </s>" over whitespace-split words; unknown words map to <unk>. */
  encode(text) {
    const unk = this.vocab[this.unk];
    const ids = text.split(/\s+/).filter(Boolean).map((w) => this.vocab[w] ?? unk);
    return [this.bos, ...ids, this.eos];
  }

  stats() {
    return { type: "WordLevel", vocabSize: this.id2tok.size, special: [...this.special].sort((a, b) => a - b), bos: this.bos, eos: this.eos, pad: this.pad };
  }
}
