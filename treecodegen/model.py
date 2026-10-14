"""Encoder (BERT), node embedding, and tree decoder.

The encoder reads the prompt. The decoder reads the tree written so far, one entry per token, and at each
step scores the next move: a father, a line under it, and a token. See EXPLANATION.md for a walk-through.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """First (2+max_anc)*k_slot dims: path (self | father | ancestors, nearest first), one slot per level.
    Rest: token + in-line position.

    Each slot = ID code (who the node is) + child-number vector (where it sits among its siblings).
    Every ID, in any role, is looked up in the same table (id_code): fixed random unit vectors, never trained.
    Child numbers are learned, one table per level. The level is given by the slot; missing ancestors are zeros,
    and so is the root's child number (it has none).
    """

    def __init__(self, d_model, vocab_size, n_ids, k_slot, max_anc, max_pos, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Row i is the code of ID i. The extra last row (index n_ids) means "no node": the root's father.
        # A buffer, not a parameter: IDs are meaningless symbols, so their codes are never trained.
        self.register_buffer("id_code", F.normalize(torch.randn(n_ids + 1, k_slot, generator=g), dim=-1))
        # Row r of rank[level] is the vector for "child number r" at that level (a node has < n_ids children).
        self.rank = nn.Parameter(0.02 * torch.randn(2 + max_anc, n_ids, k_slot, generator=g))
        k_path = (2 + max_anc) * k_slot
        if d_model - k_path < 64:
            raise ValueError(f"d_model={d_model} leaves {d_model - k_path} dims for token + position "
                             f"after the path ({k_path} = (2+max_anc)*k_slot); raise d_model")
        self.tok = nn.Embedding(vocab_size, d_model - k_path)
        self.pos = nn.Embedding(max_pos, d_model - k_path)
        self.path_norm = nn.LayerNorm(k_path)
        self.tok_norm = nn.LayerNorm(d_model - k_path)

    def forward(self, self_id, fa_id, anc, rank, tok, pos):
        """All [B,L] except anc [B,L,max_anc] and rank [B,L,2+max_anc] (-1 = none) -> [B,L,d_model]."""
        codes = self.id_code[anc.clamp(min=0)] * (anc >= 0).unsqueeze(-1)  # [B,L,max_anc,k_slot]
        ids = torch.cat([self.id_code[self_id][:, :, None], self.id_code[fa_id][:, :, None], codes], 2)
        level = torch.arange(rank.shape[-1], device=rank.device)
        ranks = self.rank[level, rank.clamp(min=0)] * (rank >= 0).unsqueeze(-1)  # [B,L,2+max_anc,k_slot]
        path = (ids + ranks).flatten(2)
        return torch.cat([self.path_norm(path), self.tok_norm(self.tok(tok) + self.pos(pos))], -1)


class PromptEncoder(nn.Module):
    """Pretrained BERT if a name is given, else a small BERT trained from scratch."""

    def __init__(self, bert_name=None, vocab_size=None, hidden=128, layers=2, heads=4):
        super().__init__()
        from transformers import BertConfig, BertModel
        # "eager" attention: the fused one doesn't support dropout on Apple GPUs (mps); prompts are short anyway
        if bert_name:
            self.bert = BertModel.from_pretrained(bert_name, add_pooling_layer=False, attn_implementation="eager")
        else:
            cfg = BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                             num_attention_heads=heads, intermediate_size=4 * hidden, pad_token_id=0, attn_implementation="eager")
            self.bert = BertModel(cfg, add_pooling_layer=False)
        self.hidden = self.bert.config.hidden_size

    def forward(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state


class TreeDecoder(nn.Module):
    """Causal transformer decoder with no sequence position encoding.

    Each step makes three choices, each head seeing the earlier ones:
      father_logp() -> the father: any existing node (scored against its ID code), or STOP
      id_logp()     -> an existing child of that father (extend its line), or a new line
      token_logits() -> the token, added to the end of that line
    The father and the line are passed on as the hidden state of their latest entry.
    """

    def __init__(self, vocab_size, d_enc, d_model=576, n_layers=4, n_heads=4, dropout=0.1,
                 n_ids=512, k_slot=64, max_anc=4, max_pos=128):
        super().__init__()
        self.d = d_model
        self.emb = NodeEmbedding(d_model, vocab_size, n_ids, k_slot, max_anc, max_pos)
        self.mem_proj = nn.Linear(d_enc, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.layers = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        # 1. father: a query compared with each existing node's ID code; one extra score for STOP
        self.fa_q, self.stop = nn.Linear(d_model, k_slot), nn.Linear(d_model, 1)
        self.fa_cond = nn.Linear(d_model, d_model)       # adds the chosen father's state to h
        # 2. line: a query compared with the ID codes of the father's children; one extra score for a new line
        self.id_q, self.new = nn.Linear(d_model, k_slot), nn.Linear(d_model, 1)
        self.line_cond = nn.Linear(d_model, d_model)     # adds the chosen line's state ...
        self.new_line = nn.Parameter(torch.zeros(d_model))  # ... or this vector for a new line
        # 3. token: over the code vocabulary
        self.tok_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))

    def forward(self, b, memory, mem_pad):
        """Entries [B,L] + prompt memory [B,P,d_enc] -> hidden states H [B,L,d_model].

        H[:, t] has seen entries 0..t (causal mask) and the whole prompt (cross-attention).
        It is used to predict move t+1. Order comes only from the child numbers and in-line positions.
        """
        x = self.emb(b["self_id"], b["fa_id"], b["anc"], b["rank"], b["tok"], b["pos"])
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        return self.layers(x, self.mem_proj(memory), tgt_mask=causal, tgt_key_padding_mask=b["pad"],
                           memory_key_padding_mask=mem_pad)

    def father_logp(self, H, node_ids, exists):
        """H [B,L,d], node_ids [B,N], exists [B,L,N] -> log P(father) [B,L,N+1]; the last column is STOP."""
        s = torch.einsum("blk,bnk->bln", self.fa_q(H), self.emb.id_code[node_ids])
        return torch.cat([s.masked_fill(~exists, float("-inf")), self.stop(H)], -1).log_softmax(-1)

    def with_father(self, h, h_father):
        """Condition on the chosen father: h_father is the state of the father's latest entry."""
        return h + self.fa_cond(h_father)

    def id_logp(self, u, node_ids, allowed):
        """u [M,d], node_ids [M,N], allowed [M,N] -> log P(line) [M,N+1]; the last column is a new line."""
        s = (self.id_q(u) @ self.emb.id_code.T).gather(1, node_ids)
        return torch.cat([s.masked_fill(~allowed, float("-inf")), self.new(u)], -1).log_softmax(-1)

    def token_logits(self, u, h_line, existing):
        """existing [M] bool: extending a line (use its latest entry) or starting a new one."""
        return self.tok_head(u + torch.where(existing[:, None], self.line_cond(h_line), self.new_line))


def build_models(cfg, vocab_size, prompt_vocab_size):
    enc = PromptEncoder(cfg["bert"], prompt_vocab_size, cfg["enc_hidden"], cfg["enc_layers"])
    dec = TreeDecoder(vocab_size, enc.hidden, cfg["d_model"], cfg["layers"], cfg["heads"], cfg["dropout"],
                      cfg["n_ids"], cfg["k_slot"], cfg["max_anc"], cfg["max_pos"])
    return enc, dec
