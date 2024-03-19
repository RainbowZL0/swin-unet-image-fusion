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
            x, y = [put_channel_dim_to_the_last_position(elem) for elem in [x, y]]
            x, y = self.norm_layer_1(x), self.norm_layer_2(y)
            x, y = [put_channel_dim_to_the_second_position(elem) for elem in [x, y]]
            x, y = self.other_module.forward(x, y)
            return short_cut_x + x, short_cut_y + y
        else:
            short_cut_x = x
            x = put_channel_dim_to_the_last_position(x)
            x = self.norm_layer_1(x)
            x = put_channel_dim_to_the_second_position(x)
            x = self.other_module.forward(x=x, y=None)
            return short_cut_x + x
