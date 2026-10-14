"""Frontier loss: -log of the total probability of the valid next moves."""
import torch


def group_logsumexp(x, g, G):
    """logsumexp of x within each group g (values 0..G-1)."""
    m = torch.full((G,), float("-inf"), device=x.device).scatter_reduce(0, g, x, "amax", include_self=True)
    s = torch.zeros(G, device=x.device).scatter_add(0, g, (x - m[g]).exp())
    return m + s.log()


def compute_loss(dec, H, b, real_weight=1.0):
    """Returns (loss, mean loss at the real points).

    A move is (father, line, token): log P = log P(father) + log P(line | father) + log P(token | father, line).
    Each step's loss is -log of the summed probability of its valid moves (plus STOP at the end).
    """
    B, L, _ = H.shape
    t = torch.arange(L, device=H.device)
    N = b["node_ids"].shape[1]
    # Father: at step t, only nodes that exist by then (first entry <= t) can be chosen.
    P_fa = dec.father_logp(H, b["node_ids"], b["node_first"][:, None, :] <= t[None, :, None])

    # Each recorded move (one row of b["moves"]) gets its own father-conditioned state u.
    mb, mt, mf, mj, mtok, mfe, mje = b["moves"].unbind(1)
    u = dec.with_father(H[mb, mt], H[mb, mfe])
    allowed = (b["node_parent"][mb] == mf[:, None]) & (b["node_first"][mb] <= mt[:, None])  # father's children so far
    P_id = dec.id_logp(u, b["node_ids"][mb], allowed)
    P_tok = dec.token_logits(u, H[mb, mje.clamp(min=0)], mj >= 0).log_softmax(-1)
    terms = (P_fa[mb, mt, mf] + P_id[torch.arange(len(mb), device=H.device), torch.where(mj >= 0, mj, N)]
             + P_tok.gather(1, mtok[:, None]).squeeze(1))

    # STOP is a valid move once per sequence: after its last entry.
    sb, st = b["stop"].unbind(1)
    terms = torch.cat([terms, P_fa[sb, st, N]])
    # Sum the probabilities of all valid moves at the same step (logsumexp per (b, t) group).
    keep = ~b["pad"].flatten()                         # one group per real entry; padding gets none
    gid = torch.full((B * L,), -1, dtype=torch.long, device=H.device)
    gid[keep] = torch.arange(int(keep.sum()), device=H.device)
    step = -group_logsumexp(terms, gid[torch.cat([mb * L + mt, sb * L + st])], int(keep.sum()))

    # Average over steps; the real point of each sequence counts real_weight times.
    w = torch.ones_like(step)
    if len(b["real"]):
        rb, rt = b["real"].unbind(1)
        r = gid[rb * L + rt]
        w[r] = real_weight
        real = step[r].mean().item()
    else:
        real = float("nan")
    return (step * w).sum() / w.sum(), real
