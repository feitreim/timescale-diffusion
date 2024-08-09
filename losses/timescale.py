"""
PyTorch implementation of custom timescale-related loss functions.
"""

__author__ = "Joshua Kovac <kovacj@wwu.edu>"

# External imports
from typing import Union, List
import torch

# Global Contants
MAX_LATENT_SIZE = 512


# ------------------------------ Timescale Loss -------------------------------
class TimescaleLoss(torch.nn.Module):
    """
    Timescale based distance metrics to enforce "closeness" of latent spaces on
    shared timescales.

    Args:
        - bin_size (int): The uniform size of each timescale "bin."
        - offset (int): The timescale offset. First <offset> elements of the vector get lumped into smallest timescale.
        - normalization (str | List[int], optional): The per-element normalization weights as a string type, or given list values
        - reduction (str, optional): The reduction method for going for an n-D tensor to a scalar.
        - distance_metric (str, optional): The distance metric for comparing latent vectors.
    """

    def __init__(
        self,
        latent_size: int,
        bin_size: int,
        offset: int = 0,
        normalization: Union[str, List[int]] = "exp",
        reduction: str = "mean",
        distance_metric: str = "contrastive",
    ) -> None:
        super(TimescaleLoss, self).__init__()
        self.bin_size = bin_size
        self.offset = offset
        self.reduction = reduction
        self.distance_metric = distance_metric
        if normalization == "exp":
            self.normalization = torch.tensor(
                [
                    2 ** ((i + 1) / self.bin_size) - 2 ** (i / self.bin_size)
                    for i in range(latent_size)
                ]
            )
            self.normalization /= torch.mean(self.normalization)
        else:
            self.normalization = torch.tensor([1.0 for _ in range(latent_size)])
        assert self.reduction in [
            "mean",
            "sum",
            "none",
        ], "LOSS INTIIALIZATION ERROR: Unknown reduction type requested."
        assert self.distance_metric in [
            "l1",
            "l2",
            "contrastive",
        ], "LOSS INTIIALIZATION ERROR: Unknown distance metric type requested."
        assert (
            self.offset >= 0
        ), "LOSS INITIALIZATION ERROR: Offset must be non-negative"

    def timescale_loss_l1(
        self, latents: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates an L1-distance based timescale loss.

        Args:
            - latents (torch.Tensor): The batch in (B, latent_dim) or (B, latent_dim, H, W)
            - time_steps (torch.Tensor): The time_steps of each image in the batch (B)

        Returns:
            - loss (torch.Tensor): The loss in a torch.Tensor object (for gradient purposes)
        """
        # Calculate time offset_matrix
        time_offset_matrix = time_steps.repeat(time_steps.size()[0], 1)
        time_offset_matrix = (
            torch.abs(time_offset_matrix - time_offset_matrix.t()) + self.offset
        )
        # Calculate bin start indices
        bin_start_indices = torch.ceil(
            torch.log2(time_offset_matrix) * self.bin_size
        ).to(torch.int32)  # BxB TODO: Switch to match pyramid levels?
        bin_start_indices = torch.clamp(
            bin_start_indices, min=0, max=(latents.size()[1])
        )
        # Calculate losses TODO: Vectorize this?
        distances = torch.zeros_like(bin_start_indices, dtype=torch.float32)
        for i in range(distances.size()[0]):
            for j in range(distances.size()[0]):
                if bin_start_indices[i, j] < latents.size()[1]:
                    if latents.dim() == 2:
                        distance = torch.nn.functional.l1_loss(
                            latents[j, bin_start_indices[i, j] :],
                            latents[i, bin_start_indices[i, j] :],
                            reduction="none",
                        )
                        distance /= (
                            self.normalization[
                                bin_start_indices[i, j] : latents.size(1)
                            ]
                            / self.normalization[latents.size(1) - 1]
                        ).to(distance.device)
                    elif latents.dim() == 4:
                        distance = torch.nn.functional.l1_loss(
                            latents[j, bin_start_indices[i, j] :, :, :],
                            latents[i, bin_start_indices[i, j] :, :, :],
                            reduction="none",
                        )
                        distance /= (
                            self.normalization[
                                bin_start_indices[i, j] : latents.size(1)
                            ].view(-1, 1, 1)
                            / self.normalization[latents.size(1) - 1]
                        ).to(distance.device)
                    else:
                        assert False, f"LOSS ERROR: X must have 2 or 4 dimensions, had {latents.dim()}."
                    distances[i, j] = torch.sum(distance)
        # Reduction
        if self.reduction == "mean":
            loss = torch.mean(distances)
        if self.reduction == "sum":
            loss = torch.sum(distances)
        return loss

    def timescale_loss_l2(
        self, latents: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates an L2-distance based timescale loss.

        Args:
            - latents (torch.Tensor): The batch in (B, latent_dim) or (B, latent_dim, H, W)
            - time_steps (torch.Tensor): The time_steps of each image in the batch (B)

        Returns:
            - loss (torch.Tensor): The loss in a torch.Tensor object (for gradient purposes)
        """
        # Calculate time offset_matrix
        time_offset_matrix = time_steps.repeat(time_steps.size()[0], 1)
        time_offset_matrix = (
            torch.abs(time_offset_matrix - time_offset_matrix.t()) + self.offset
        )
        # Calculate bin start indices
        bin_start_indices = torch.ceil(
            torch.log2(time_offset_matrix) * self.bin_size
        ).to(torch.int32)  # BxB TODO: Switch to match pyramid levels?
        bin_start_indices = torch.clamp(
            bin_start_indices, min=0, max=(latents.size()[1])
        )
        # Calculate losses TODO: Vectorize this?
        distances = torch.zeros_like(bin_start_indices, dtype=torch.float32)
        for i in range(distances.size()[0]):
            for j in range(distances.size()[0]):
                if bin_start_indices[i, j] < latents.size()[1]:
                    if latents.dim() == 2:
                        distance = torch.nn.functional.mse_loss(
                            latents[j, bin_start_indices[i, j] :],
                            latents[i, bin_start_indices[i, j] :],
                            reduction="none",
                        )
                        distance /= (
                            self.normalization[
                                bin_start_indices[i, j] : latents.size(1)
                            ]
                            / self.normalization[latents.size(1) - 1]
                        ).to(distance.device)
                    elif latents.dim() == 4:
                        distance = torch.nn.functional.mse_loss(
                            latents[j, bin_start_indices[i, j] :, :, :],
                            latents[i, bin_start_indices[i, j] :, :, :],
                            reduction="none",
                        )
                        distance /= (
                            self.normalization[
                                bin_start_indices[i, j] : latents.size(1)
                            ].view(-1, 1, 1)
                            / self.normalization[latents.size(1) - 1]
                        ).to(distance.device)
                    else:
                        assert False, f"LOSS ERROR: X must have 2 or 4 dimensions, had {latents.dim()}."
                    distances[i, j] = torch.sum(distance)
        # Reduction
        if self.reduction == "mean":
            loss = torch.mean(distances)
        if self.reduction == "sum":
            loss = torch.sum(distances)
        return loss

    def timescale_loss_contrastive(
        self, latents: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates a timescale loss similar to SimCLR's contrastive loss but with temporal dependencies in the "sim" function.

        Args:
            - latents (torch.Tensor): The batch in (B, latent_dim) or (B, latent_dim, H, W)
            - time_steps (torch.Tensor): The time_steps of each image in the batch (B)

        Returns:
            - loss (torch.Tensor): The loss in a torch.Tensor object (for gradient purposes)
        """
        B = latents.size()[0]  # Batch size
        L = latents.size()[1]
        losses = torch.zeros((B, B), dtype=torch.float32)
        # Calculate time offset_matrix
        print(f"latents:{latents.shape}, time_steps:{time_steps.shape}")
        time_offset_matrix = time_steps.repeat(time_steps.size()[0], 1)
        time_offset_matrix = (
            torch.abs(time_offset_matrix - time_offset_matrix.t()) + self.offset
        )
        # Calculate bin start indices
        bin_start_indices = torch.ceil(
            torch.log2(time_offset_matrix) * self.bin_size
        ).to(torch.int32)  # BxB TODO: Switch to match pyramid levels?
        bin_start_indices = torch.clamp(bin_start_indices, min=0, max=L)
        # Calculate the numerators
        numerators = torch.unsqueeze(latents, dim=0).repeat((B, 1, 1, 1, 1))
        for i in range(B):
            for j in range(B):
                numerators[i, j, : bin_start_indices[i, j]] = 0
        numerators = torch.flatten(numerators, start_dim=2)
        numerators = torch.nn.functional.normalize(numerators, dim=2)
        numerators = numerators * torch.transpose(numerators, 0, 1)
        numerators = torch.sum(numerators, axis=2)
        numerators = torch.exp(numerators)
        # Calculate the denominators and subsequent losses
        positive_pair_count = 0
        for i in range(B):
            for j in range(B):
                if i == j or bin_start_indices[i, j] >= L:
                    losses[i, j] = 0.0
                else:
                    denominators_ij = latents.clone()
                    denominators_ij[:, : bin_start_indices[i, j]] = 0
                    denominators_ij = torch.flatten(denominators_ij, start_dim=1)
                    denominators_ij = torch.nn.functional.normalize(
                        denominators_ij, dim=1
                    )
                    denominator_ij = (
                        torch.sum(denominators_ij * denominators_ij[i]) - 1.0
                    )
                    losses[i, j] = numerators[i, j] / denominator_ij
                    positive_pair_count += 1
        # Reduction
        if self.reduction == "mean":
            loss = torch.sum(losses) / positive_pair_count
        if self.reduction == "sum":
            loss = torch.sum(losses)
        return loss

    def forward(self, latents: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - latents (torch.Tensor): The batch in (B, latent_dim) or (B, latent_dim, H, W)
            - time_steps (torch.Tensor): The time_steps of each image in the batch (B)

        Returns:
            - loss (torch.Tensor): The loss in a torch.Tensor object (for gradient purposes)
        """
        if self.distance_metric == "l1":
            return self.timescale_loss_l1(latents, time_steps)
        elif self.distance_metric == "l2":
            return self.timescale_loss_l2(latents, time_steps)
        elif self.distance_metric == "contrastive":
            return self.timescale_loss_contrastive(latents, time_steps)
        else:
            assert self.distance_metric in [
                "l1",
                "l2",
                "contrastive",
            ], "LOSS INTIIALIZATION ERROR: Unknown distance metric type requested."


def periodic_loss(z, t, w, w2):
    r"""
    z: [B C H W]
    t: [B T] T=7
    w: weight of shape: [B T H 1]
    w2: [B T 1 W]
    """
    _, T = t.shape
    inter = z[:, :T] @ w

    return torch.pow((z[:, :T] @ w) - t, 2.0).mean()
