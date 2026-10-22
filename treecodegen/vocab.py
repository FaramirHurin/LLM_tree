"""Vocabularies: code tokens for the decoder, prompt tokens for the encoder."""
import re
from collections import Counter

PAD, UNK, ROOT, END = "<pad>", "<unk>", "<root>", "<end>"


class Vocab:
    """Code-token vocabulary for the decoder."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, trees, min_freq=1, max_size=None):
        """Tokens seen at least min_freq times; with max_size, only the most frequent ones, so that the
        vocabulary (the 4 special tokens included) has at most max_size entries. The rest become <unk>."""
        cnt = Counter()
        stack = list(trees)
        while stack:
            n = stack.pop()
            cnt.update(n.content)
            stack.extend(n.children)
        specials = [PAD, UNK, ROOT, END]
        kept = [t for t, c in cnt.items() if c >= min_freq]
        if max_size is not None:
            kept = sorted(kept, key=lambda t: (-cnt[t], t))[: max(0, max_size - len(specials))]
        vocab = cls(specials + sorted(kept))
        vocab.coverage = sum(cnt[t] for t in kept) / max(1, sum(cnt.values()))   # share of tokens not <unk>
        return vocab

    def __len__(self):
        return len(self.itos)

    def id(self, t):
        return self.stoi.get(t, self.stoi[UNK])


class PromptTokenizer:
    """Pretrained BERT tokenizer if a model name is given, else a simple word tokenizer."""

    def __init__(self, bert_name=None, texts=None, itos=None, max_len=128):
        self.bert_name, self.max_len = bert_name, max_len
        if bert_name:
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained(bert_name)
            self.vocab_size = self.hf.vocab_size
        else:
            if itos is None:
                cnt = Counter(w for t in texts for w in self._words(t))
                itos = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted(cnt)
            self.itos = itos
            self.stoi = {w: i for i, w in enumerate(itos)}
            self.vocab_size = len(itos)

    @staticmethod
    def _words(text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def __call__(self, texts):
        import torch
        if self.bert_name:
            b = self.hf(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            return b["input_ids"], b["attention_mask"]
        rows = [[2] + [self.stoi.get(w, 1) for w in self._words(t)][: self.max_len - 2] + [3] for t in texts]
        L = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (L - len(r)) for r in rows])
        return ids, (ids != 0).long()
