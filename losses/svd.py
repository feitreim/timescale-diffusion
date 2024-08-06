import torch


def singular_value_loss(x, t):
    u, s, vh = torch.svd(x)
    t_x = t.unsqueeze(1).expand(t.shape[0], s.shape[1], 7)
    return (s[:, :, :7] - t_x).mean()
