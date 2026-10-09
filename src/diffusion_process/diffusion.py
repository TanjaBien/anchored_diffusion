import torch

from src.diffusion_process.schedule import create_diffusion_schedule, extract


class DenoisingDiffusion:
    """The noise schedule and the ancestral (DDPM) sampler of an x0-predicting model."""

    def __init__(self, n_steps, device, snr_gamma=5.0):
        self.n_steps = n_steps
        self.device = device
        self.diff_dict = create_diffusion_schedule(n_steps, device, snr_gamma)

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Draw samples of the given shape with all n_steps reverse steps; no noise in the last."""
        x = torch.randn(shape, device=self.diff_dict['alphas'].device)
        for i in reversed(range(self.n_steps)):
            t = torch.tensor([i], device=x.device)
            x0 = model(x, t.repeat(len(x)))
            mean = (extract(self.diff_dict['posterior_mean_coef1'], t, x) * x0
                    + extract(self.diff_dict['posterior_mean_coef2'], t, x) * x)
            z = torch.randn_like(x)
            sigma = extract(self.diff_dict['posterior_variance'], t, x).sqrt()
            x = mean + (1.0 - (t == 0).float()) * sigma * z
        return x
