import torch


#est, ref : (B, T). Retourne un SI-SDR en dB par élément du batch.
#    Plus c'est grand, mieux c'est. Invariant au gain de l'estimation.
def si_sdr(est, ref, eps=1e-8):
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    # Projection de l'estimation sur la référence : la "bonne" partie du signal
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10(proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)
