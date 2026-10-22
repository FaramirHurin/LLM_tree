"""Encoder (BERT), node embedding, and tree decoder.

The encoder reads the prompt. The decoder reads the tree written so far, one entry per token, and at each
step scores the next move: a father, a line under it (for a new line, also its place among the siblings),
and a token. See EXPLANATION.md for a walk-through.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """First (2+max_anc+3)*k_slot + k_depth dims: path (self | father | ancestors, nearest first, one slot per
    level | left 1 | left 2 | right 1 | depth). Rest: content = token + in-line position.

    Each ID slot holds an ID code. Every ID, in any role, is looked up in the same table (id_code): fixed random
    unit vectors, never trained. The neighbour slots hold the line's siblings when the entry was written (EDGE
    past the first or last one), which is how the entries encode sibling order. Missing slots are zeros.
    The depth slot is learned, one vector per depth (the root is 0; depths >= max_depth - 1 share the last row).
    The ID slots alone can't tell depths apart once every ancestor slot is full.
    """

    def __init__(self, d_model, vocab_size, n_ids, k_slot, max_anc, max_pos, max_depth, k_depth, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Row i is the code of ID i. Two extra rows: n_ids = "no node" (the root's father), n_ids + 1 = EDGE
        # (past the first or last sibling). A buffer, not a parameter: IDs are meaningless symbols, never trained.
        self.register_buffer("id_code", F.normalize(torch.randn(n_ids + 2, k_slot, generator=g), dim=-1))
        k_path = (2 + max_anc + 3) * k_slot + k_depth
        if d_model - k_path < 64:
            raise ValueError(f"d_model={d_model} leaves {d_model - k_path} dims for token + position "
                             f"after the path ({k_path} = (2+max_anc+3)*k_slot + k_depth); raise d_model")
        self.depth = nn.Embedding(max_depth, k_depth)
        self.tok = nn.Embedding(vocab_size, d_model - k_path)
        self.pos = nn.Embedding(max_pos, d_model - k_path)
        self.path_norm = nn.LayerNorm(k_path)
        self.tok_norm = nn.LayerNorm(d_model - k_path)

    def forward(self, self_id, fa_id, anc, nb, tok, pos, depth):
        """All [B,L] except anc [B,L,max_anc] and nb [B,L,3] (-1 = none) -> [B,L,d_model]."""
        slots = torch.cat([self_id[..., None], fa_id[..., None], anc, nb], -1)          # [B,L,2+max_anc+3]
        ids = self.id_code[slots.clamp(min=0)] * (slots >= 0).unsqueeze(-1)              # missing slots are zeros
        depth = self.depth(depth.clamp(max=self.depth.num_embeddings - 1))               # [B,L,k_depth]
        path = torch.cat([ids.flatten(2), depth], -1)
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

    Each step makes up to four choices, each head seeing the earlier ones:
      father_logp() -> the father: any existing node (scored against its ID code), or STOP
      id_logp()     -> an open child of that father (extend its line), or a new line
      place_logp()  -> for a new line: the existing child it goes right after, or first
      token_logits() -> the token, added to the end of that line
    The father, the line and the place are passed on as the hidden state of their latest entry.
    """

    def __init__(self, vocab_size, d_enc, d_model=896, n_layers=4, n_heads=4, dropout=0.1,
                 n_ids=512, k_slot=64, max_anc=4, max_pos=128, max_depth=16, k_depth=32):
        super().__init__()
        self.d = d_model
        self.emb = NodeEmbedding(d_model, vocab_size, n_ids, k_slot, max_anc, max_pos, max_depth, k_depth)
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
        # 3. place of a new line: a query compared with the ID codes of the father's children; one extra score for first
        self.place_q, self.first = nn.Linear(d_model, k_slot), nn.Linear(d_model, 1)
        self.place_cond = nn.Linear(d_model, d_model)    # adds the state of the child it goes after ...
        self.first_line = nn.Parameter(torch.zeros(d_model))  # ... or this vector when it goes first
        # 4. token: over the code vocabulary
        self.tok_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))

    def forward(self, b, memory, mem_pad):
        """Entries [B,L] + prompt memory [B,P,d_enc] -> hidden states H [B,L,d_model].

        H[:, t] has seen entries 0..t (causal mask) and the whole prompt (cross-attention).
        It is used to predict move t+1. Order comes only from the neighbours and in-line positions.
        """
        x = self.emb(b["self_id"], b["fa_id"], b["anc"], b["nb"], b["tok"], b["pos"], b["depth"])
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        return self.layers(x, self.mem_proj(memory), tgt_mask=causal, tgt_key_padding_mask=b["pad"],
                           memory_key_padding_mask=mem_pad)

    @torch.no_grad()
    def extend(self, b, memory, mem_pad, cache):
        """Incremental forward for generation: b holds only the entries added since the last call ([1, n]).

        cache (a dict, empty at first) keeps each layer's self-attention keys and values, the prompt's keys and
        values, and every hidden state so far; returns H for all entries [1, L, d_model]. It gives the same H as
        forward() on all entries: an entry never changes once written, and attention is causal. Eval mode only
        (no dropout); it reuses the layers' own weights.
        """
        x = self.emb(b["self_id"], b["fa_id"], b["anc"], b["nb"], b["tok"], b["pos"], b["depth"])
        n = x.shape[1]
        if not cache:
            cache.update(k=[], v=[], mk=[], mv=[], H=x.new_zeros(1, 0, self.d), mem=self.mem_proj(memory))
        L0 = cache["H"].shape[1]
        allow = torch.ones(n, L0 + n, dtype=torch.bool, device=x.device).tril(L0)   # new entry i sees 0..L0+i
        mem_allow = (~mem_pad)[:, None, None, :]
        for i, layer in enumerate(self.layers.layers):
            sa, ca = layer.self_attn, layer.multihead_attn
            split = lambda t: t.view(1, -1, sa.num_heads, self.d // sa.num_heads).transpose(1, 2)
            merge = lambda t: t.transpose(1, 2).reshape(1, n, self.d)
            # self-attention: the new entries' keys and values join the cache
            q, k, v = F.linear(layer.norm1(x), sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
            if len(cache["k"]) <= i:
                cache["k"].append(split(k)); cache["v"].append(split(v))
            else:
                cache["k"][i] = torch.cat([cache["k"][i], split(k)], 2)
                cache["v"][i] = torch.cat([cache["v"][i], split(v)], 2)
            a = F.scaled_dot_product_attention(split(q), cache["k"][i], cache["v"][i], attn_mask=allow)
            x = x + sa.out_proj(merge(a))
            # cross-attention to the prompt: its keys and values are computed once
            wq, wk, wv = ca.in_proj_weight.chunk(3)
            bq, bk, bv = ca.in_proj_bias.chunk(3)
            if len(cache["mk"]) <= i:
                cache["mk"].append(split(F.linear(cache["mem"], wk, bk)))
                cache["mv"].append(split(F.linear(cache["mem"], wv, bv)))
            q = split(F.linear(layer.norm2(x), wq, bq))
            a = F.scaled_dot_product_attention(q, cache["mk"][i], cache["mv"][i], attn_mask=mem_allow)
            x = x + ca.out_proj(merge(a))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        cache["H"] = torch.cat([cache["H"], self.layers.norm(x)], 1)
        return cache["H"]

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

    def place_logp(self, u, node_ids, allowed):
        """For a new line: u [M,d], node_ids [M,N], allowed [M,N] (the father's existing children)
        -> log P(goes right after that child) [M,N+1]; the last column is "first"."""
        q = u + self.new_line
        s = (self.place_q(q) @ self.emb.id_code.T).gather(1, node_ids)
        return torch.cat([s.masked_fill(~allowed, float("-inf")), self.first(q)], -1).log_softmax(-1)

    def token_logits(self, u, h_line, existing, h_place, has_place):
        """existing [M] bool: extending a line (use its latest entry) or starting a new one, which is placed
        right after a sibling (has_place: use that sibling's latest entry h_place) or first."""
        new = self.new_line + torch.where(has_place[:, None], self.place_cond(h_place), self.first_line)
        return self.tok_head(u + torch.where(existing[:, None], self.line_cond(h_line), new))


def build_models(cfg, vocab_size, prompt_vocab_size):
    enc = PromptEncoder(cfg["bert"], prompt_vocab_size, cfg["enc_hidden"], cfg["enc_layers"])
    dec = TreeDecoder(vocab_size, enc.hidden, cfg["d_model"], cfg["layers"], cfg["heads"], cfg["dropout"],
                      cfg["n_ids"], cfg["k_slot"], cfg["max_anc"], cfg["max_pos"], cfg.get("max_depth", 16),
                      cfg.get("k_depth", 32))
    return enc, dec
