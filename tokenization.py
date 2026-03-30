"""
Part 3: Tokenization Gap
Replaces WordPiece with BPE and Character-level tokenization,
evaluates performance, and implements extensions.
"""

import re
import time
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────
# 1.  BASE TOKENIZER INTERFACE
# ─────────────────────────────────────────────

class BaseTokenizer:
    """Common interface for all tokenizers."""

    def train(self, corpus: List[str]) -> None:
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab.get("[UNK]", 0)) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        id2tok = {v: k for k, v in self.vocab.items()}
        return " ".join(id2tok.get(i, "[UNK]") for i in ids)


# ─────────────────────────────────────────────
# 2.  WORDPIECE TOKENIZER  (baseline)
# ─────────────────────────────────────────────

class WordPieceTokenizer(BaseTokenizer):
    """Simplified WordPiece – greedy longest-match from the left."""

    SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}

    def train(self, corpus: List[str]) -> None:
        # Build character-level vocabulary, then greedily add whole words
        char_freq: Counter = Counter()
        word_freq: Counter = Counter()
        for text in corpus:
            for word in text.lower().split():
                word_freq[word] += 1
                for ch in word:
                    char_freq[ch] += 1

        vocab_set = set(self.SPECIAL)
        # Add all individual characters first
        for ch in char_freq:
            vocab_set.add(ch)
            vocab_set.add("##" + ch)
        # Add whole words until budget is exhausted
        for word, _ in word_freq.most_common():
            if len(vocab_set) >= self.vocab_size:
                break
            vocab_set.add(word)

        self.vocab = {tok: idx for idx, tok in enumerate(sorted(vocab_set))}

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.lower().split():
            tokens.extend(self._tokenize_word(word))
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        if word in self.vocab:
            return [word]
        sub_tokens: List[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                substr = word[start:end]
                candidate = substr if start == 0 else "##" + substr
                if candidate in self.vocab:
                    found = candidate
                    break
                end -= 1
            if found is None:
                return ["[UNK]"]
            sub_tokens.append(found)
            start = end
        return sub_tokens


# ─────────────────────────────────────────────
# 3.  BYTE-PAIR ENCODING (BPE) TOKENIZER
# ─────────────────────────────────────────────

class BPETokenizer(BaseTokenizer):
    """
    Full BPE implementation:
      1. Start with character-level vocabulary.
      2. Iteratively merge the most frequent adjacent pair.
      3. Store merge rules for inference.
    """

    def __init__(self, vocab_size: int = 500, num_merges: Optional[int] = None):
        self.vocab_size = vocab_size
        self.num_merges = num_merges  # if None, run until vocab_size is reached
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}

    # ── Training ──────────────────────────────

    @staticmethod
    def _get_vocab(corpus: List[str]) -> Dict[str, int]:
        """Represent each word as space-separated characters + end-of-word marker."""
        vocab: Dict[str, int] = Counter()
        for text in corpus:
            for word in text.lower().split():
                vocab[" ".join(list(word)) + " </w>"] += 1
        return dict(vocab)

    @staticmethod
    def _get_stats(vocab: Dict[str, int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        new_vocab: Dict[str, int] = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        replacement = "".join(pair)
        for word, freq in vocab.items():
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, corpus: List[str]) -> None:
        vocab = self._get_vocab(corpus)

        # Collect initial character-level tokens
        token_set: set = set()
        for word in vocab:
            for sym in word.split():
                token_set.add(sym)

        max_merges = self.num_merges or (self.vocab_size - len(token_set) - 1)

        for _ in range(max_merges):
            if len(token_set) >= self.vocab_size:
                break
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            vocab = self._merge_vocab(best, vocab)
            token_set.add("".join(best))

        # Build final vocabulary
        all_tokens = ["[PAD]", "[UNK]"] + sorted(token_set)
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}

    # ── Inference ─────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.lower().split():
            word_tokens = self._apply_merges(word)
            tokens.extend(word_tokens)
        return tokens

    def _apply_merges(self, word: str) -> List[str]:
        symbols = list(word) + ["</w>"]
        for left, right in self.merges:
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                    new_symbols.append(left + right)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return [s if s in self.vocab else "[UNK]" for s in symbols]


# ─────────────────────────────────────────────
# 4.  CHARACTER-LEVEL TOKENIZER
# ─────────────────────────────────────────────

class CharacterTokenizer(BaseTokenizer):
    """Splits every token into individual characters."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}

    def train(self, corpus: List[str]) -> None:
        char_set: set = {"[PAD]", "[UNK]", "[SPACE]"}
        for text in corpus:
            for ch in text.lower():
                char_set.add(ch)
        self.vocab = {ch: idx for idx, ch in enumerate(sorted(char_set))}

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for ch in text.lower():
            if ch == " ":
                tokens.append("[SPACE]")
            elif ch in self.vocab:
                tokens.append(ch)
            else:
                tokens.append("[UNK]")
        return tokens


# ─────────────────────────────────────────────
# 5.  EXTENSION A – Hybrid Tokenizer (word + character)
# ─────────────────────────────────────────────

class HybridTokenizer(BaseTokenizer):
    """
    Uses whole-word tokens for high-frequency words;
    falls back to character-level for rare/OOV words.
    """

    def __init__(self, word_vocab_size: int = 300, freq_threshold: int = 3):
        self.word_vocab_size = word_vocab_size
        self.freq_threshold = freq_threshold
        self.word_vocab: set = set()
        self.vocab: Dict[str, int] = {}

    def train(self, corpus: List[str]) -> None:
        word_freq: Counter = Counter()
        char_set: set = set()
        for text in corpus:
            for word in text.lower().split():
                word_freq[word] += 1
                for ch in word:
                    char_set.add(ch)

        # High-frequency words enter the word vocabulary
        self.word_vocab = {
            word for word, freq in word_freq.most_common(self.word_vocab_size)
            if freq >= self.freq_threshold
        }

        special = {"[PAD]", "[UNK]", "[SPACE]"}
        all_tokens = special | self.word_vocab | char_set | {"##" + ch for ch in char_set}
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.lower().split():
            if word in self.word_vocab:
                tokens.append(word)
            else:
                # Character-level fallback
                for i, ch in enumerate(word):
                    tokens.append(ch if i == 0 else "##" + ch)
        return tokens


# ─────────────────────────────────────────────
# 6.  EXTENSION B – Dynamic Token Merging
# ─────────────────────────────────────────────

class DynamicMergingTokenizer(BaseTokenizer):
    """
    Like BPE but re-evaluates merge priorities on a sliding window of
    the *current* corpus being tokenized (online / adaptive merging).
    """

    def __init__(self, base_merges: int = 200, dynamic_merges: int = 50):
        self.base_merges = base_merges
        self.dynamic_merges = dynamic_merges
        self.bpe = BPETokenizer(num_merges=base_merges)
        self.vocab: Dict[str, int] = {}
        self.extra_merges: List[Tuple[str, str]] = []

    def train(self, corpus: List[str]) -> None:
        self.bpe.train(corpus)
        self.vocab = dict(self.bpe.vocab)

    def adapt(self, domain_corpus: List[str]) -> None:
        """
        Learn additional merges from a domain-specific corpus
        and update vocabulary accordingly.
        """
        # Tokenize with base BPE, then find frequent adjacent pairs
        tokenized = [self.bpe.tokenize(text) for text in domain_corpus]
        pair_freq: Counter = Counter()
        for seq in tokenized:
            for a, b in zip(seq, seq[1:]):
                pair_freq[(a, b)] += 1

        new_tokens: List[str] = []
        for (a, b), _ in pair_freq.most_common(self.dynamic_merges):
            merged = a + b
            if merged not in self.vocab:
                self.extra_merges.append((a, b))
                new_tokens.append(merged)

        next_id = max(self.vocab.values()) + 1
        for tok in new_tokens:
            self.vocab[tok] = next_id
            next_id += 1
        self.bpe.vocab = self.vocab  # share updated vocab

    def tokenize(self, text: str) -> List[str]:
        tokens = self.bpe.tokenize(text)
        # Apply extra dynamic merges
        for left, right in self.extra_merges:
            merged_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    merged_tokens.append(left + right)
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1
            tokens = merged_tokens
        return tokens


# ─────────────────────────────────────────────
# 7.  PERFORMANCE EVALUATOR
# ─────────────────────────────────────────────

class TokenizerEvaluator:
    """Measures vocabulary size, sequence compression, OOV rate, and speed."""

    def __init__(self, corpus: List[str], test_texts: List[str]):
        self.corpus = corpus
        self.test_texts = test_texts

    def evaluate(self, tokenizer: BaseTokenizer, name: str) -> Dict:
        # Train
        t0 = time.perf_counter()
        tokenizer.train(self.corpus)
        train_time = time.perf_counter() - t0

        # Tokenize test set
        t1 = time.perf_counter()
        all_tokens: List[List[str]] = [tokenizer.tokenize(t) for t in self.test_texts]
        infer_time = time.perf_counter() - t1

        # Metrics
        total_chars = sum(len(t) for t in self.test_texts)
        total_tokens = sum(len(seq) for seq in all_tokens)
        oov_count = sum(seq.count("[UNK]") for seq in all_tokens)
        oov_rate = oov_count / max(total_tokens, 1)
        compression = total_chars / max(total_tokens, 1)

        results = {
            "tokenizer": name,
            "vocab_size": len(tokenizer.vocab),
            "avg_tokens_per_sentence": round(total_tokens / len(self.test_texts), 2),
            "compression_ratio": round(compression, 3),  # chars per token
            "oov_rate": round(oov_rate, 4),
            "train_time_s": round(train_time, 4),
            "infer_time_s": round(infer_time, 4),
        }
        return results

    def compare_all(self, tokenizers: Dict[str, BaseTokenizer]) -> List[Dict]:
        results = []
        for name, tok in tokenizers.items():
            r = self.evaluate(tok, name)
            results.append(r)
        return results


# ─────────────────────────────────────────────
# 8.  DEMO
# ─────────────────────────────────────────────

def run_demo():
    # ── Small corpus ──
    CORPUS = [
        "the quick brown fox jumps over the lazy dog",
        "natural language processing is a subfield of artificial intelligence",
        "tokenization is the process of splitting text into tokens",
        "byte pair encoding merges frequent character pairs iteratively",
        "character level models can handle any out of vocabulary word",
        "transformers use wordpiece or byte pair encoding for tokenization",
        "the model learns representations of subword units",
        "pre-training on large corpora improves downstream task performance",
        "attention mechanisms allow the model to focus on relevant tokens",
        "fine-tuning adapts a pre-trained model to a specific task",
    ]

    TEST = [
        "the fox and the dog are friends",
        "tokenization improves model performance significantly",
        "unseen words like hyperparametrization are tricky",
    ]

    print("=" * 65)
    print("  PART 3: TOKENIZATION GAP COMPARISON")
    print("=" * 65)

    # ── Instantiate tokenizers ──
    wp  = WordPieceTokenizer(vocab_size=500)
    bpe = BPETokenizer(vocab_size=500, num_merges=100)
    ch  = CharacterTokenizer()
    hyb = HybridTokenizer(word_vocab_size=200, freq_threshold=2)
    dyn = DynamicMergingTokenizer(base_merges=80, dynamic_merges=20)

    tokenizers = {
        "WordPiece (baseline)": wp,
        "BPE": bpe,
        "Character-level": ch,
        "Hybrid (word+char)": hyb,
        "Dynamic Merging": dyn,
    }

    # ── Evaluate ──
    evaluator = TokenizerEvaluator(CORPUS, TEST)
    results = evaluator.compare_all(tokenizers)

    # ── Print table ──
    cols = ["tokenizer", "vocab_size", "avg_tokens_per_sentence",
            "compression_ratio", "oov_rate", "train_time_s", "infer_time_s"]
    headers = ["Tokenizer", "Vocab", "Avg Tokens", "Chars/Token", "OOV Rate",
               "Train(s)", "Infer(s)"]
    widths = [24, 7, 11, 12, 10, 9, 9]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + header_line)
    print("-" * len(header_line))
    for r in results:
        row = "  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths))
        print(row)

    # ── Sample tokenizations ──
    print("\n" + "=" * 65)
    print("  SAMPLE TOKENIZATIONS")
    print("=" * 65)
    sample = "tokenization improves model performance significantly"
    for name, tok in tokenizers.items():
        toks = tok.tokenize(sample)
        print(f"\n[{name}]")
        print(f"  {toks}")

    # ── Adaptive domain corpus for Dynamic Merging ──
    print("\n" + "=" * 65)
    print("  DYNAMIC MERGING: DOMAIN ADAPTATION")
    print("=" * 65)
    domain_corpus = [
        "hyperparameter tuning is crucial for model optimization",
        "backpropagation computes gradients efficiently",
        "regularization prevents overfitting in deep neural networks",
    ]
    dyn_adapted = DynamicMergingTokenizer(base_merges=80, dynamic_merges=20)
    dyn_adapted.train(CORPUS)
    before = dyn_adapted.tokenize("hyperparameter optimization")
    dyn_adapted.adapt(domain_corpus)
    after = dyn_adapted.tokenize("hyperparameter optimization")
    print(f"\nBefore adaptation: {before}")
    print(f"After  adaptation: {after}")
    print(f"\nExtra merges learned: {dyn_adapted.extra_merges[:5]}")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_demo()
