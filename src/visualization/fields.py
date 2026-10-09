"""Figures of fields and their residuals, or of shocked-circle samples."""
import numpy as np


def residual_maps(residual, field_shape):
    """|R| of the first residual channel as images (n, n0, n1)."""
    residual = residual.detach().float().cpu()
    if residual.ndim == 3:
        residual = residual[..., 0]
    return residual.abs().reshape(len(residual), *field_shape).numpy()


def sample_figure(samples, residual=None, field_shape=None, title='', count=4):
    """The first `count` samples, one column each: every channel, then |R| if given.

    Shocked-circle samples (n, D) are drawn as points in the data plane instead.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples = samples.detach().float().cpu().numpy()
    if samples.ndim == 2:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        angle = np.linspace(0.0, 2.0 * np.pi, 720)
        ax.plot(np.cos(angle), np.sin(angle), color='0.75', lw=0.9)
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.35, color='0.25', linewidths=0)
        ax.set_aspect('equal')
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_title(f'{title} {len(samples)} samples in the data plane', fontsize=9)
        return fig

    count = min(count, len(samples))
    maps = residual_maps(residual[:count], field_shape) if residual is not None else None
    rows = samples.shape[1] + (maps is not None)
    fig, axes = plt.subplots(rows, count, figsize=(2.4 * count, 2.2 * rows), squeeze=False)
    for col in range(count):
        for channel in range(samples.shape[1]):
            image = axes[channel, col].imshow(samples[col, channel], cmap='viridis', aspect='auto')
            fig.colorbar(image, ax=axes[channel, col], fraction=0.046)
        if maps is not None:
            image = axes[-1, col].imshow(maps[col], cmap='magma', aspect='auto',
                                         vmax=np.percentile(maps[col], 99) or None)
            fig.colorbar(image, ax=axes[-1, col], fraction=0.046)
            axes[-1, col].set_title(f'|R| mean {maps[col].mean():.2e}', fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
