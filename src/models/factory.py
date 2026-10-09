from src.config import device
from src.models.toy import ToyMLP
from src.models.unet import Unet3D


def build_model(setup):
    """The x0-predicting denoiser: an MLP on the toy, the U-Net on the PDEs."""
    if setup.gov_eqs == 'toy':
        return ToyMLP(setup.channels, setup.diff_steps).to(device)
    return Unet3D(dim=32, channels=setup.channels).to(device)
