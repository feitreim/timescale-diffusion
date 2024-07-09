"""
PyTorch implementations of Reconstruction Losses.
"""

# External imports
from typing import Union, Tuple, List
import torch
from torch import Tensor
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from losses.ssim import MS_SSIM
from einops import rearrange


class L2Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        loss = y - x
        loss = loss * loss
        return loss.mean()


# --------------------------- Lp Reconstruction Loss ---------------------------
class LpReconstructionLoss(torch.nn.Module):
    """
    Image reconstruction losses with some specified Lp distance metric with
    specified p.

    Args:
        - p (float, optional): The p ordinality of the norm used as a distance metric.
        - reduction (str, optional): The method used for reducing the loss.
    """

    def __init__(self, p: Union[float, str] = 1.0, reduction: str = "mean") -> None:
        super(LpReconstructionLoss, self).__init__()
        # Validate the inputs and save for later use
        if isinstance(p, float):
            assert (
                p >= 0
            ), f"RECONSTRUCTION LOSS ERROR: p must be non-negative, was {p}."
        if isinstance(p, str):
            assert p in [
                "fro",
                "nuc",
            ], f"RECONSTRUCTION LOSS ERROR: non-ordinal p was not recognized, was {p}."
        self.p = p
        assert (
            reduction.lower()
            in [
                "mean",
                "sum",
                "none",
            ]
        ), f"RECONSTRUCTION LOSS ERROR: {reduction.lower()} is an unrecognized reduction method."
        self.reduction = reduction.lower()

    def forward(
        self, reconstructions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            - reconstructions (torch.Tensor): The image reconstructions of the shape (B, C, H, W).
            - targets (torch.Tensor): The reconstruction targets of the shape (B, C, H, W).

        Returns:
            - reconstruction_loss (torch.Tensor): The (potentially reduced) reconstruction loss.
        """
        # Validate the input shapes
        assert (
            reconstructions.size() == targets.size()
        ), f"RECONSTRUCTION LOSS ERROR: reconstructions and targets did not have the same dimensions, were {reconstructions.size()} and {targets.size()}."
        # Flatten the vectors
        reconstructions = torch.flatten(reconstructions, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)
        # Take their distances
        distances = targets - reconstructions
        # Take the lp norm of the distances
        reconstruction_losses = torch.linalg.vector_norm(distances, ord=self.p, dim=1)
        # Reduce
        if self.reduction == "mean":
            reconstruction_loss = torch.mean(reconstruction_losses)
        elif self.reduction == "sum":
            reconstruction_loss = torch.sum(reconstruction_losses)
        else:
            reconstruction_loss = reconstruction_losses
        # Retrun the resulting loss
        return reconstruction_loss


# ----------------------------- MS-SSIM + L1 Mix Loss --------------------------
class MixReconstructionLoss(torch.nn.Module):
    """
    PyTorch implementation of the MS-SSIM + L1 "Mix" Loss from "Loss Functions
    for Image Restoration with Neural Networks" by Zhao et al.

    Args:
        - win_size (int, optional): The window size of the Gaussian kernel.
        - win_sigma (float, optional): The sigma standard deviation of the Gaussian kernel.
        - weights (Tuple[float, float, float, float, float], optional): The weights for each level scale level in MS-SSIM.
        - K (Tuple[float, float], optional): The scalar constants (K1, K2) used for SSIM at each scale.
        - mix_alpha (float, optional): The alpha value for the convex combination of MS-SSIM and L1 reconstruction losses.
    """

    def __init__(
        self,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: List[float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        K: Tuple[float, float] = (0.01, 0.03),
        mix_alpha: float = 0.9,
    ) -> None:
        super(MixReconstructionLoss, self).__init__()
        self.ms_ssim = MS_SSIM(
            data_range=1.0,
            size_average=True,
            win_size=win_size,
            win_sigma=win_sigma,
            weights=weights,
            K=K,
        )
        """
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
            gaussian_kernel=True,
            kernel_size=win_size,
            sigma=win_sigma,
            reduction="elementwise_mean",
            data_range=1.0,
            k1=K[0],
            k2=K[1],
            betas=weights,
            normalize="relu",
        )
        """
        self.l1 = torch.nn.L1Loss()
        assert (
            0.0 <= mix_alpha <= 1.0
        ), f"RECONSTRUCTION LOSS ERROR: mix_alpha must be in [0, 1], was {mix_alpha}."
        self.mix_alpha = mix_alpha

    def forward(
        self, reconstructions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            - reconstructions (torch.Tensor): The image reconstructions of the shape (B, C, H, W).
            - targets (torch.Tensor): The reconstruction targets of the shape (B, C, H, W).

        Returns:
            - reconstruction_loss (torch.Tensor): The reduced reconstruction loss.
        """
        if reconstructions.ndim > 4:
            reconstructions = rearrange(reconstructions, "B T C H W -> (B T) C H W")
        if targets.ndim > 4:
            targets = rearrange(targets, "B T C H W -> (B T) C H W")

        # Calculate the MS-SSIM Loss
        ms_ssim_loss = 1 - self.ms_ssim(reconstructions, targets)
        # Calculate l1 loss
        l1_loss = self.l1(reconstructions, targets)
        # Combine into a weighted sum
        reconstruction_loss = (
            self.mix_alpha * ms_ssim_loss + (1 - self.mix_alpha) * l1_loss
        )
        return reconstruction_loss
