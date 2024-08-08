import torch


@torch.compile()
def singular_value_loss(z, t):
    """
    likely to work best if C = T
    shapes:
    x: B C H W
    t: B T

    s <- B C H
    s <- B C H[0] = B C 1

    t <-  B T 1
    """
    B, T = t.shape
    _, s, _ = torch.svd(z)

    s = s[:, :T, 0].view(B, T, 1)
    t = t.view(B, T, 1)

    return torch.pow(s - t, 2.0).mean()
