from src.methods.utils import denoise


def loss(model, diffusion, residual, x_0, config):
    """The data term alone: the diffusion model without physics (DM)."""
    _t, _x0_hat, data_loss = denoise(model, diffusion, x_0)
    data_loss = config['c_data'] * data_loss.mean()
    return data_loss, data_loss.item(), 0.0, 0.0
