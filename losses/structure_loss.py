import torch
from utils import download_artifact
import toml

from classifier import TSCVQVAE


class StructureLoss:
    """
    Let X be the complete feature space of the webcam image
                                               ┌───────┐
                                               │   X   │                              ┌───┐
           ┌───┐          ┌────────────────────┴───────┴────────────────────┐         │ x │
           │ t │          │┌─────────────────────────────┐┌────────────────┐│         └─┬─┘
           └─┬─┘          ││                             ││                ││           │
             │            ││                             ││                ││  ┌────────▼────────┐
    ┌────────▼─────────┐  ││                             ││                ││  │  Training X_u   │
    │  Pretrained X_t  │  ││                             ││                ││  │    predictor    │
    │     Decoder      │  ││                             ││                ││  └────────┬────────┘
    └────────┬─────────┘  ││                             ││                ││           │
             │            ││             X_t             ││      X_u       ││           │
             │            ││       Time-Dependent        ││     Unique     ││           │
             │            ││          Features           ││    Features    ◀┼───────────┘
             └────────────┼▶                             ││                ││
                          ││                             ││                ││
                          │└─────────────────────────────┘└────────────────┘│
                          └─────────────────────────────────────────────────┘
    """

    def __init__(self, model_artifact, device):
        state, args = download_artifact(model_artifact)
        state_dict = torch.load(state, map_location=device)
        model_args = toml.load(args)["vqvae"]
        self.model = TSCVQVAE(**model_args)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def __call__(self, t, pred, target):
        """
        calculate x_t from the pretrained model.
        then let x_u = x - x_t, then calculate
        the absolute error between x_u and pred.
        """
        x_t = self.model.generate_image_from_timecode(t)
        x_u = target - x_t
        return torch.abs(x_u - pred).mean(), pred + x_t
