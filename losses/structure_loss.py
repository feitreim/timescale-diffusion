import torch
import torch.nn.functional as f
from utils import download_artifact
import toml

from classifier import TSCVQVAE


class StructureLoss:
    """
            Let X be the complete feature space of the webcam image
                                 ┌─────────────────┐
                      ┌──────────┤X: img feat space├─────────┐
                      │          └─────────────────┘         │
                      │┌────────────────────────────────────┐│      ┌─────────┐
    ┌────────────┐    ││                                    ││      │Training │
    │ Structural ├────┼▶     X_u: unique subspace of X      ├┼──────▶ Decoder │
    │Autoencoder │    ││                                    ││      └─────────┘
    └────────────┘    │└────────────────────────────────────┘│
                      │┌────────────────────────────────────┐│
    ┌────────────┐    ││                                    ││      ┌─────────┐
    │  Temporal  ├────┼▶ X_t: time-dependant subspace of X  ├┼──────▶ Frozen  │
    │Autoencoder │    ││                                    ││      │ Decoder │
    └────────────┘    │└────────────────────────────────────┘│      └─────────┘
                      └──────────────────────────────────────┘
    """

    def __init__(self, model_artifact, device, patch_size, k):
        self.patch_size = patch_size
        self.k = k
        state, args = download_artifact(model_artifact)
        state_dict = torch.load(state, map_location=device, weights_only=False)
        model_args = toml.load(args)['vqvae']
        self.model = TSCVQVAE(**model_args).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def __call__(self, t, pred, target):
        """
        calculate x_t from the pretrained model.
        then let x_u = x - x_t, then calculate

        then we're going to calculate loss on the
        k largest magnitude patches of x_u
        """
        with torch.no_grad():
            _, x_t, _, _ = self.model.generate_image_from_timecode(t)
        x_u = target - x_t

        x_u_patches = f.unfold(x_u, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        pred_patches = f.unfold(pred, kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        x_u_mag = x_u_patches.abs().sum(dim=1, keepdim=True)
        patch_indices = torch.argsort(x_u_mag, dim=2, descending=True)[:, 0 : self.k]
        biggest_x_u = torch.take_along_dim(x_u_patches, patch_indices, dim=2)
        match_pred_patches = torch.take_along_dim(pred_patches, patch_indices, dim=2)
        difference = biggest_x_u - match_pred_patches
        return torch.pow(difference, 2.0).mean(), pred + x_t
