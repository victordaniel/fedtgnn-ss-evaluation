"""FedTGNN-SS training (proposed method) with switchable components for
ablation. All training uses only each silo's local training graph."""

import copy
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from .graph import insert_eval_nodes, knn_graph
from .models import FedTGNNEncoder, fedavg, load_avg, n_params_bytes
from .heads import federated_lr_head


@dataclass
class FedTGNNConfig:
    hidden: int = 128          # embedding dim = hidden // 2 = 64
    dropout: float = 0.4
    lr: float = 0.005
    weight_decay: float = 5e-4
    rounds: int = 10           # T
    local_epochs: int = 3      # R
    k: int = 10                # initial k-NN
    k_agr: int = 10            # k-NN for adaptive graph refinement
    agr_every: int = 5         # K (federation rounds)
    tau0: float = 0.90
    tau_decay: float = 0.03    # lambda
    tau_min: float = 0.70
    consensus: float = 0.5
    blend: float = 0.5         # local/global prototype blend
    eta: float = 0.8           # pseudo-label loss weight
    lam_smooth: float = 0.05
    lam_contra: float = 0.1
    lam_aug: float = 0.3
    tau_c: float = 0.5
    noise_std: float = 0.05
    mu_prox: float = 0.01
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    # component switches
    federated: bool = True
    use_pl: bool = True
    pl_mode: str = 'pgpl'      # 'pgpl' (3 gates) or 'conf' (gate 1 only)
    use_agr: bool = True
    use_caa: bool = True
    use_proto_share: bool = True
    use_focal: bool = True
    use_contra: bool = True
    use_smooth: bool = True
    use_prox: bool = True
    edge_attention: bool = True
    head: str = 'fedlr'        # 'fedlr' (calibrated federated LR on [X||H]) or 'gnn'
    # evaluation shortcuts, used ONLY in the leakage experiment
    leak_pooled_head: bool = False   # L2: final LR fitted on pooled data of all silos
    leak_global_graph: bool = False  # L4: one cross-silo graph incl. eval patients at inference


# ------------------------------------------------------------------ losses
def focal_loss(logits, y, alpha, gamma, weights=None):
    """Binary focal loss on the softmax probability of the true class:
    FL = -a_t (1 - p_t)^gamma log p_t,  a_t = alpha if y=1 else 1-alpha."""
    logp = F.log_softmax(logits, 1).gather(1, y.view(-1, 1)).squeeze(1)
    pt = logp.exp()
    at = torch.where(y == 1, torch.full_like(pt, alpha), torch.full_like(pt, 1 - alpha))
    fl = -at * (1 - pt) ** gamma * logp
    if weights is not None:
        return (fl * weights).sum() / weights.sum().clamp(min=1e-8)
    return fl.mean()


def ce_loss(logits, y, weights=None):
    l = F.cross_entropy(logits, y, reduction='none')
    if weights is not None:
        return (l * weights).sum() / weights.sum().clamp(min=1e-8)
    return l.mean()


def supcon_loss(h, y, tau):
    if len(y) < 4 or len(torch.unique(y)) < 2:
        return h.sum() * 0
    z = F.normalize(h, dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = F.log_softmax(sim, 1)
    return (-(logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def smooth_loss(h, ei, pred):
    same = pred[ei[0]] == pred[ei[1]]
    if same.sum() == 0:
        return h.sum() * 0
    return (h[ei[0][same]] - h[ei[1][same]]).pow(2).sum(1).mean()


# ------------------------------------------------------ pseudo-labelling
def prototypes(h, y, mask):
    return {c: h[mask & (y == c)].mean(0) for c in (0, 1)
            if (mask & (y == c)).sum() > 0}


def blend_protos(local, glob, b):
    if not glob:
        return local
    out = {}
    for c in (0, 1):
        if c in local and c in glob:
            out[c] = b * local[c] + (1 - b) * glob[c]
        elif c in local:
            out[c] = local[c]
        elif c in glob:
            out[c] = glob[c]
    return out


def pseudo_label(logits, h, ei, cand_mask, protos, tau, cfg):
    """Returns (mask, labels, weights) for newly pseudo-labelled nodes."""
    p = F.softmax(logits, 1)
    conf, pred = p.max(1)
    gate = cand_mask & (conf >= tau)
    w = conf.clone()
    if cfg.pl_mode == 'pgpl':
        if len(protos) < 2:
            return torch.zeros_like(gate), pred, w * 0
        d0 = (h - protos[0]).norm(dim=1)
        d1 = (h - protos[1]).norm(dim=1)
        d_pred = torch.where(pred == 0, d0, d1)
        d_oth = torch.where(pred == 0, d1, d0)
        gate &= d_pred < d_oth                                   # gate 2
        src, dst = ei
        agree = torch.zeros(len(pred)).index_add_(
            0, dst, (pred[src] == pred[dst]).float())
        deg = torch.zeros(len(pred)).index_add_(0, dst, torch.ones(len(dst)))
        gate &= (agree / deg.clamp(min=1) >= cfg.consensus) & (deg > 0)  # gate 3
        w = conf * (1 - d_pred / (d_pred + d_oth + 1e-8))
    return gate, pred, w


# ------------------------------------------------------------- inference
def _forward_eval(model, silo, Z_eval, graph_space, agr_sigma):
    """Inductive inference for evaluation patients at one silo."""
    model.eval()
    n_tr = len(silo.X)
    with torch.no_grad():
        ei, ew = insert_eval_nodes(silo.X.numpy(), Z_eval.numpy(), silo.k,
                                   silo.feat_sigma, silo.ei, silo.ew)
        x = torch.cat([silo.X, Z_eval])
        logits, h = model(x, ei, ew)
        if graph_space == 'emb':
            # two-step insertion: neighbours re-found in embedding space
            H_tr, H_ev = h[:n_tr], h[n_tr:]
            ei, ew = insert_eval_nodes(H_tr.numpy(), H_ev.numpy(), silo.k,
                                       agr_sigma, silo.ei, silo.ew)
            logits, h = model(x, ei, ew)
    return logits[n_tr:], h[n_tr:], h[:n_tr]


def _global_graph_inference(model, silos, k):
    """L4 shortcut: a single symmetric k-NN graph over every silo's training
    patients and every evaluation patient (cross-silo edges, eval patients
    connected to each other). Returns head_data in the usual format."""
    blocks, spans = [], []
    for s in silos:
        spans.append(('train', s, len(s.X)))
        blocks.append(s.X)
    for name in ('val', 'test'):
        for s in silos:
            spans.append((name, s, len(s.eval_X[name])))
            blocks.append(s.eval_X[name])
    Z = torch.cat(blocks)
    ei, ew, _ = knn_graph(Z.numpy(), k)
    model.eval()
    with torch.no_grad():
        lg, h = model(Z, ei, ew)
    pos, parts = 0, {}
    for kind, s, n in spans:
        parts[(kind, s.sid)] = (lg[pos:pos + n], h[pos:pos + n])
        pos += n
    return [(s, parts[('train', s.sid)][1], {nm: parts[(nm, s.sid)] for nm in ('val', 'test')})
            for s in silos]


# ------------------------------------------------------------------ train
def train_fedtgnn(ctx, cfg=FedTGNNConfig(), seed=0, diag=None, graph_hook=None):
    """Train FedTGNN-SS on a FoldContext. Returns dict name -> (idx, prob)
    for 'val' and 'test', plus cost statistics in '_meta'.
    graph_hook(silo, edge_index, stage) is called with stage='initial' and
    stage='refined' (diagnostic use only; it cannot affect training)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    silos = [copy.copy(s) for s in ctx.silos]   # AGR must not alter ctx
    for s in silos:
        s.k, s.feat_sigma = cfg.k, s.sigma
        s.feat_ei, s.feat_ew = s.ei, s.ew
        s.space, s.agr_sigma = 'feat', None
        s.y_work = s.y.clone()
        s.w_pl = torch.zeros(len(s.y))
        s.is_pl = torch.zeros(len(s.y), dtype=torch.bool)
        if graph_hook is not None:
            graph_hook(s, s.ei, 'initial')
    model0 = FedTGNNEncoder(ctx.d, cfg.hidden, cfg.dropout, cfg.edge_attention)
    models = [copy.deepcopy(model0) for _ in silos]
    g_state = copy.deepcopy(model0.state_dict())
    g_protos = {}
    comm_bytes = 0
    pbytes = n_params_bytes(model0)
    loss_fn = ((lambda lg, y, w=None: focal_loss(lg, y, cfg.focal_alpha, cfg.focal_gamma, w))
               if cfg.use_focal else ce_loss)
    sizes = [len(s.X) for s in silos]

    for t in range(cfg.rounds):
        tau = max(cfg.tau0 * math.exp(-cfg.tau_decay * t), cfg.tau_min)
        states, protos_up, lab_counts = [], [], []
        for s, m in zip(silos, models):
            if cfg.federated:
                m.load_state_dict(g_state)
            g_params = {k: v.clone() for k, v in m.named_parameters()}
            opt = torch.optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for _ in range(cfg.local_epochs):
                m.train()
                opt.zero_grad()
                logits, h = m(s.X, s.ei, s.ew)
                loss = loss_fn(logits[s.lab], s.y[s.lab])
                if cfg.use_pl and s.is_pl.any():
                    loss = loss + cfg.eta * loss_fn(logits[s.is_pl], s.y_work[s.is_pl],
                                                    s.w_pl[s.is_pl])
                if cfg.use_smooth:
                    loss = loss + cfg.lam_smooth * smooth_loss(h, s.ei, logits.argmax(1).detach())
                if cfg.use_contra:
                    loss = loss + cfg.lam_contra * supcon_loss(h[s.lab], s.y[s.lab], cfg.tau_c)
                if cfg.use_caa and s.unl.any():
                    x_aug = s.X + torch.randn_like(s.X) * cfg.noise_std * ctx.cont_mask
                    lg_aug, _ = m(x_aug, s.ei, s.ew)
                    p = F.softmax(logits[s.unl].detach(), 1)
                    loss = loss + cfg.lam_aug * F.kl_div(
                        F.log_softmax(lg_aug[s.unl], 1), p, reduction='batchmean')
                if cfg.federated and cfg.use_prox and t > 0:
                    loss = loss + cfg.mu_prox / 2 * sum(
                        (p_ - g_params[n]).pow(2).sum() for n, p_ in m.named_parameters())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

            m.eval()
            with torch.no_grad():
                logits, h = m(s.X, s.ei, s.ew)
                loc_protos = prototypes(h, s.y, s.lab)
                if cfg.use_pl:
                    use_g = g_protos if (cfg.federated and cfg.use_proto_share) else {}
                    pr = blend_protos(loc_protos, use_g, cfg.blend)
                    cand = s.unl & ~s.is_pl
                    gate, pred, w = pseudo_label(logits, h, s.ei, cand, pr, tau, cfg)
                    s.is_pl |= gate
                    s.y_work[gate] = pred[gate]
                    s.w_pl[gate] = w[gate]
                    if diag is not None:
                        pl = s.is_pl
                        diag.append(dict(round=t + 1, silo=s.sid, n_unlabeled=int(s.unl.sum()),
                                         n_pseudo=int(pl.sum()),
                                         n_correct=int((s.y_work[pl] == s.y_hidden[pl]).sum()),
                                         n_pseudo_pos=int((s.y_work[pl] == 1).sum())))
                # adaptive graph refinement (built on training patients only)
                if cfg.use_agr and (t + 1) % cfg.agr_every == 0 and (t + 1) < cfg.rounds:
                    s.ei, s.ew, s.agr_sigma = knn_graph(h.numpy(), cfg.k_agr)
                    if graph_hook is not None:
                        graph_hook(s, s.ei, 'refined')
                    s.space, s.k = 'emb', cfg.k_agr
            states.append(copy.deepcopy(m.state_dict()))
            protos_up.append(loc_protos)
            lab_counts.append({c: int((s.lab & (s.y == c)).sum()) for c in (0, 1)})

        if cfg.federated:
            g_state = fedavg(states, sizes)
            comm_bytes += 2 * pbytes * len(silos)
            if cfg.use_proto_share:
                g_protos = {}
                for c in (0, 1):
                    num = sum(lc[c] * p[c] for p, lc in zip(protos_up, lab_counts) if c in p)
                    den = sum(lc[c] for p, lc in zip(protos_up, lab_counts) if c in p)
                    if den > 0:
                        g_protos[c] = num / den
                comm_bytes += 2 * len(silos) * 2 * (cfg.hidden // 2) * 4

    if cfg.federated:
        for m in models:
            load_avg(m, g_state)

    # ---- inference at each silo (inductive)
    out = {'val': ([], []), 'test': ([], [])}
    head_data = []
    if cfg.leak_global_graph:
        # L4 (leakage experiment only): one k-NN graph over all silos'
        # training patients AND all evaluation patients, as in the original code
        head_data = _global_graph_inference(models[0], silos, cfg.k)
    else:
        for s, m in zip(silos, models):
            ev = {}
            if getattr(s, 'eval_pos', None) is not None:
                # L1 (leakage experiment only): eval patients are already nodes
                m.eval()
                with torch.no_grad():
                    lg_all, h_all = m(s.X, s.ei, s.ew)
                for name in ('val', 'test'):
                    ev[name] = (lg_all[s.eval_pos[name]], h_all[s.eval_pos[name]])
                h_tr = h_all
            else:
                for name in ('val', 'test'):
                    lg, h_ev, h_tr = _forward_eval(m, s, s.eval_X[name], s.space, s.agr_sigma)
                    ev[name] = (lg, h_ev)
            head_data.append((s, h_tr, ev))

    if cfg.head == 'fedlr':
        probs = federated_lr_head(head_data, federated=cfg.federated, seed=seed,
                                  pooled=cfg.leak_pooled_head)
        if cfg.federated:
            comm_bytes += probs.pop('_comm')
        else:
            probs.pop('_comm')
    else:
        probs = {n: [F.softmax(ev[n][0], 1)[:, 1].numpy() for _, _, ev in head_data]
                 for n in ('val', 'test')}
    for n in ('val', 'test'):
        idx = np.concatenate([s.eval_idx[n] for s in silos])
        out[n] = (idx, np.concatenate(probs[n]))
    if 'members' in probs:
        out['members'] = (np.concatenate([s.idx[s.lab.numpy()] for s in silos]),
                          np.concatenate(probs['members']))
    out['_meta'] = dict(train_time_s=time.time() - t0, comm_bytes=comm_bytes,
                        n_pseudo=int(sum(s.is_pl.sum() for s in silos)))
    return out
