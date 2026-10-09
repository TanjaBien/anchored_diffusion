import torch
import torch.nn.functional as F


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def cosine_beta_schedule(n_timesteps):
    s = 0.008
    x = torch.linspace(0, n_timesteps, n_timesteps + 1)
    alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def create_diffusion_schedule(n_steps, device, snr_gamma=5.0):
    diff_dict = {
        'betas': cosine_beta_schedule(n_steps).to(device),
    }
    diff_dict['alphas'] = 1.0 - diff_dict['betas']
    diff_dict['alphas_prod'] = torch.cumprod(diff_dict['alphas'], 0)
    diff_dict['alphas_bar_sqrt'] = torch.sqrt(diff_dict['alphas_prod'])
    diff_dict['one_minus_alphas_bar_sqrt'] = torch.sqrt(1 - diff_dict['alphas_prod'])
    diff_dict['alphas_prod_prev'] = F.pad(diff_dict['alphas_prod'][:-1], (1, 0), value=1.0)
    diff_dict['posterior_mean_coef1'] = (
        diff_dict['betas'] * torch.sqrt(diff_dict['alphas_prod_prev']) / (1.0 - diff_dict['alphas_prod'])
    )
    diff_dict['posterior_mean_coef2'] = (
        (1.0 - diff_dict['alphas_prod_prev']) * torch.sqrt(diff_dict['alphas']) / (1.0 - diff_dict['alphas_prod'])
    )
    diff_dict['posterior_variance'] = (
        diff_dict['betas'] * (1.0 - diff_dict['alphas_prod_prev']) / (1.0 - diff_dict['alphas_prod'])
    )
    diff_dict['posterior_variance_clipped'] = diff_dict['posterior_variance'].clone()
    diff_dict['posterior_variance_clipped'][0] = diff_dict['posterior_variance'][1]

    snr = diff_dict['alphas_prod'] / (1.0 - diff_dict['alphas_prod'])
    # the data term's weight lambda_t = min(SNR_t, gamma): min-SNR weighting (gamma = 5 on the PDEs)
    diff_dict['p2_loss_weight'] = torch.minimum(snr, torch.ones_like(snr) * snr_gamma)
    return diff_dict
