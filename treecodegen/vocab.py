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
    def build(cls, trees, min_freq=1):
        cnt = Counter()
        stack = list(trees)
        while stack:
            n = stack.pop()
            cnt.update(n.content)
            stack.extend(n.children)
        return cls([PAD, UNK, ROOT, END] + sorted(t for t, c in cnt.items() if c >= min_freq))

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
