import torch

from src.diffusion_process import extract


def sample_timesteps(batch_size, n_steps, device):
    """Antithetic noise levels: t and n_steps - 1 - t in pairs."""
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,), device=device)
    return torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()


def denoise(model, diffusion, x_0):
    """Noise a clean batch at random levels t and predict it back.

    Returns t, the prediction x0_hat and the data term per sample, lambda_t * mean((x0 - x0_hat)^2)
    with lambda_t = min(SNR_t, gamma).
    """
    t = sample_timesteps(len(x_0), diffusion.n_steps, x_0.device)
    noise = torch.randn_like(x_0, device=x_0.device)
    x_t = (x_0 * extract(diffusion.diff_dict['alphas_bar_sqrt'], t, x_0)
           + noise * extract(diffusion.diff_dict['one_minus_alphas_bar_sqrt'], t, x_0))
    x0_hat = model(x_t, t)
    data_loss = (x_0 - x0_hat).pow(2).reshape(len(x_0), -1).mean(dim=1)
    data_loss = data_loss * extract(diffusion.diff_dict['p2_loss_weight'], t, data_loss)
    return t, x0_hat, data_loss
