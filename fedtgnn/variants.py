"""Ablation variants and sensitivity sweeps of FedTGNN-SS."""

from dataclasses import replace

from .fedtgnn import FedTGNNConfig

BASE = FedTGNNConfig()

ABLATIONS = {
    'Full': BASE,
    'w/o PG-PL gates 2-3 (confidence only)': replace(BASE, pl_mode='conf'),
    'w/o pseudo-labelling': replace(BASE, use_pl=False),
    'w/o adaptive graph refinement': replace(BASE, use_agr=False),
    'w/o clinical-aware augmentation': replace(BASE, use_caa=False),
    'w/o prototype sharing': replace(BASE, use_proto_share=False),
    'w/o focal loss (cross-entropy)': replace(BASE, use_focal=False),
    'w/o contrastive loss': replace(BASE, use_contra=False),
    'w/o class-aware smoothness': replace(BASE, use_smooth=False),
    'w/o dynamic edge attention': replace(BASE, edge_attention=False),
    'w/o FedProx term': replace(BASE, use_prox=False),
    'w/o calibrated LR head (GNN logits)': replace(BASE, head='gnn'),
    'Supervised only (no PL, CAA, proto)': replace(
        BASE, use_pl=False, use_caa=False, use_proto_share=False),
}

SWEEPS = {
    'tau0': [0.80, 0.85, 0.90, 0.95],
    'tau_decay': [0.0, 0.03, 0.10],
    'k': [5, 10, 15, 20],
    'agr_every': [2, 3, 5],
    'blend': [0.0, 0.25, 0.5, 0.75, 1.0],
    'noise_std': [0.01, 0.05, 0.10],
    'eta': [0.2, 0.5, 0.8, 1.0],
    'rounds': [5, 10, 20],
}


def sweep_configs():
    for p, vals in SWEEPS.items():
        for v in vals:
            kw = {p: v}
            if p == 'k':
                kw['k_agr'] = v
            yield f'{p}={v}', replace(BASE, **kw)


# ---- leakage experiment (exploratory; PROTOCOL.md Amendment 2) -------------
# (name, FedTGNNConfig or None, build_context kwargs, baseline method or None)
LEAKAGE = [
    ('Correct protocol (this study)', BASE, {}, None),
    ('L1: test patients in training graph', BASE, {'transductive': True}, None),
    ('L2: pooled final classifier', replace(BASE, leak_pooled_head=True), {}, None),
    ('L3: preprocessing fitted on all data', BASE, {'leak_preprocess': True}, None),
    ('L4: one cross-silo graph at inference', replace(BASE, leak_global_graph=True), {}, None),
    ('All four shortcuts', replace(BASE, leak_pooled_head=True, leak_global_graph=True),
     {'transductive': True, 'leak_preprocess': True}, None),
    ('FedAvg-LR, correct protocol', None, {}, 'FedAvg-LR'),
    ('FedAvg-LR, L3', None, {'leak_preprocess': True}, 'FedAvg-LR'),
]
