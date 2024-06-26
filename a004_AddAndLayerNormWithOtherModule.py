from torch import nn, Tensor
from a007_utils import *


class AddAndLayerNormWithOtherModule(nn.Module):
    def __init__(
        self, normalized_shape: list, use_dual_path: bool, other_module: nn.Module
    ):
        super().__init__()
        """copy args to attributes"""
        self.normalized_shape = normalized_shape
        self.use_dual_path = use_dual_path
        self.other_module = other_module

        """new attributes"""
        self.norm_layer_1 = nn.LayerNorm(normalized_shape=normalized_shape)
        if use_dual_path:
            self.norm_layer_2 = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x, y):
        """
        Args:
            x (Tensor): shape 4D, (batch_size, channels, h, w)
            y (Tensor | None)

        Returns:
            1 or 2 tensors, shape 4D, (batch_size, channels, h, w)
        """
        if self.use_dual_path or y is not None:
            short_cut_x, short_cut_y = x, y
            x, y = my_layer_norm(
                x=x,
                layer_x=self.norm_layer_1,
                y=y,
                layer_y=self.norm_layer_2
            )
            x, y = self.other_module(x, y)
            return short_cut_x + x, short_cut_y + y
        else:
            short_cut_x = x
            x = my_layer_norm(
                x=x,
                layer_x=self.norm_layer_1,
                y=None,
                layer_y=None,
            )
            x = self.other_module(x=x, y=None)
            return short_cut_x + x

    def forward_(self, x, y):
        return self(x, y)


def my_layer_norm(x, layer_x, y=None, layer_y=None):
    """
    Args:
        x: (b c h w)
        layer_x: nn.Module
        y: (b c h w)
        layer_y: nn.Module
    Returns: after layer norm
    """
    if y is not None:
        x, y = [put_channel_dim_to_the_last_position(elem) for elem in [x, y]]
        x, y = layer_x(x), layer_y(y)
        x, y = [put_channel_dim_to_the_second_position(elem) for elem in [x, y]]
        return x, y
    else:
        x = put_channel_dim_to_the_last_position(x)
        x = layer_x(x)
        x = put_channel_dim_to_the_second_position(x)
        return x
