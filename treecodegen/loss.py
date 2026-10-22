"""Frontier loss: -log of the total probability of the valid next moves."""
import torch


def group_logsumexp(x, g, G):
    """logsumexp of x within each group g (values 0..G-1)."""
    m = torch.full((G,), float("-inf"), device=x.device).scatter_reduce(0, g, x, "amax", include_self=True)
    s = torch.zeros(G, device=x.device).scatter_add(0, g, (x - m[g]).exp())
    return m + s.log()


def compute_loss(dec, H, b, real_weight=1.0):
    """Returns (loss, mean loss at the real points, floor-adjusted loss).

    The floor-adjusted loss is for reading only (training uses the loss): a step with k distinct valid moves, of
    which m are scored, can't go below log(k / m) even for a perfect model, since the model isn't told which m are
    scored. Subtracting that floor at each step gives a number that a perfect model brings to about 0.

    A move is (father, line, [place,] token): log P = log P(father) + log P(line | father)
    + [for a new line: log P(place | father)] + log P(token | father, line, place).
    Each step's loss is -log of the summed probability of its valid moves (plus STOP at the end).
    """
    B, L, _ = H.shape
    t = torch.arange(L, device=H.device)
    N = b["node_ids"].shape[1]
    # Father: at step t, only nodes that exist by then (first entry <= t) can be chosen.
    P_fa = dec.father_logp(H, b["node_ids"], b["node_first"][:, None, :] <= t[None, :, None])

    # Each recorded move (one row of b["moves"]) gets its own father-conditioned state u.
    mb, mt, mf, mj, mtok, mfe, mje, mpl, mple = b["moves"].unbind(1)
    rows = torch.arange(len(mb), device=H.device)
    u = dec.with_father(H[mb, mt], H[mb, mfe])
    # the father's children that exist by then; lines can only extend those not closed yet (<end> still to come)
    exist = (b["node_parent"][mb] == mf[:, None]) & (b["node_first"][mb] <= mt[:, None])
    P_id = dec.id_logp(u, b["node_ids"][mb], exist & (b["node_end"][mb] > mt[:, None]))
    # place of a new line: right after one of those children (closed ones too), or first
    new = mj < 0
    P_pl = dec.place_logp(u, b["node_ids"][mb], exist)[rows, torch.where(mpl >= 0, mpl, N)]
    P_tok = dec.token_logits(u, H[mb, mje.clamp(min=0)], ~new, H[mb, mple.clamp(min=0)], mpl >= 0).log_softmax(-1)
    terms = (P_fa[mb, mt, mf] + P_id[rows, torch.where(new, N, mj)] + torch.where(new, P_pl, 0.0)
             + P_tok.gather(1, mtok[:, None]).squeeze(1))

    # STOP is a valid move once per sequence: after its last entry.
    sb, st = b["stop"].unbind(1)
    terms = torch.cat([terms, P_fa[sb, st, N]])
    # Sum the probabilities of all valid moves at the same step (logsumexp per (b, t) group).
    keep = ~b["pad"].flatten()                         # one group per real entry; padding gets none
    G = int(keep.sum())
    gid = torch.full((B * L,), -1, dtype=torch.long, device=H.device)
    gid[keep] = torch.arange(G, device=H.device)
    groups = gid[torch.cat([mb * L + mt, sb * L + st])]
    step = -group_logsumexp(terms, groups, G)
    scored = torch.bincount(groups, minlength=G).float()
    floor = torch.log(b["n_valid"].flatten()[keep].float() / scored)

    # Average over steps; the real point of each sequence counts real_weight times.
    w = torch.ones_like(step)
    if len(b["real"]):
        rb, rt = b["real"].unbind(1)
        r = gid[rb * L + rt]
        w[r] = real_weight
        real = step[r].mean().item()
    else:
        real = float("nan")
    norm = (((step - floor) * w).sum() / w.sum()).item()
    return (step * w).sum() / w.sum(), real, norm
