from einops import rearrange


def generalized_image_to_b_xy_c(tensor):
    """[batch, channels..., x, y] -> [batch, x*y, channels...]."""
    channel_axes = ' '.join(f'c{i}' for i in range(tensor.ndim - 3))
    return rearrange(tensor, f'b {channel_axes} x y -> b (x y) {channel_axes}')


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """[batch, x*y, channels...] -> [batch, channels..., x, y]; square unless the sizes are given."""
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(round(tensor.shape[1] ** 0.5))
    channel_axes = ' '.join(f'c{i}' for i in range(tensor.ndim - 2))
    return rearrange(tensor, f'b (x y) {channel_axes} -> b {channel_axes} x y', x=pixels_x, y=pixels_y)
