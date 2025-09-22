import torch
import torch.nn.functional as F

def mambalrp_silu(x):
    """MambaLRP SiLU modification"""
    identity = x.clone()
    silu_output = F.silu(x)
    return identity * (silu_output / (identity + 1e-8)).detach()

def mambalrp_multiplicative_gate(a, b):
    """MambaLRP multiplicative gate modification"""
    product = a * b
    return 0.5 * product + 0.5 * product.detach()