from torch import nn
from a007_utils import *


class AutoPathMLP(nn.Module):
    def __init__(
        self,
        in_out_dims: int,
        hidden_dims: int,
        activation_func: nn.Module,
        use_dual_path: bool,
        drop_ratio: float,
    ):
        super().__init__()
        self.in_out_dims = in_out_dims
        self.hidden_dims = hidden_dims
        self.activation_func = activation_func
        self.use_dual_path = use_dual_path
        self.drop_ratio = drop_ratio

        self.mlp_x_1 = nn.Conv2d(in_channels=in_out_dims, out_channels=hidden_dims, kernel_size=1)
        self.mlp_x_2 = nn.Conv2d(in_channels=hidden_dims, out_channels=in_out_dims, kernel_size=1)
        self.dropout_x_1 = nn.Dropout(p=drop_ratio)
        self.dropout_x_2 = nn.Dropout(p=drop_ratio)
        self.sequence_x = nn.Sequential(
            self.mlp_x_1,
            self.activation_func,
            self.dropout_x_1,
            self.mlp_x_2,
            self.dropout_x_2,
        )

        if self.use_dual_path:
            self.mlp_y_1 = nn.Conv2d(in_channels=in_out_dims, out_channels=hidden_dims, kernel_size=1)
            self.mlp_y_2 = nn.Conv2d(in_channels=hidden_dims, out_channels=in_out_dims, kernel_size=1)
            self.dropout_y_1 = nn.Dropout(p=drop_ratio)
            self.dropout_y_2 = nn.Dropout(p=drop_ratio)
            self.sequence_y = nn.Sequential(
                self.mlp_y_1,
                self.activation_func,
                self.dropout_y_1,
                self.mlp_y_2,
                self.dropout_y_2,
            )

    def forward(self, x, y):
        if self.use_dual_path or y is not None:
            return self.sequence_x(x), self.sequence_y(y)
        else:
            return self.sequence_x(x)

