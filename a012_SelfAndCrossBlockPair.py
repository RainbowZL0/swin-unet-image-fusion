import torch

from a009_NormalAndShiftWinsBlockPair import NormalAndShiftWinsBlockPair
from torch import nn


class SelfAndCrossBlockPair(nn.Module):
    """Contains self-attention and cross-attention blocks, each has one normal window and one shifted-window blocks."""

    def __init__(
        self,
        # the following params are copied from NormalAndShiftWinsBlockPair class __init__().
        # except that the "use_cross_attention" is removed, because it is determined here, not from outside.
        in_out_dims: int,
        num_heads: int,
        dims_per_head: int,
        window_size: tuple,
        use_dual_path: bool,
        use_qkv_bias: bool,
        attention_drop_ratio: float,
        linear_after_att_drop_ratio: float,
        mlp_hidden_dims: int,
        mlp_activation_func: nn.Module,
        mlp_drop_ratio: float,
    ):
        super().__init__()
        self.in_out_dims = in_out_dims
        self.num_heads = num_heads
        self.dims_per_head = dims_per_head
        self.window_size = window_size
        self.use_dual_path = use_dual_path
        self.use_qkv_bias = use_qkv_bias
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation_func = mlp_activation_func
        self.mlp_drop_ratio = mlp_drop_ratio

        # self-attention block
        self.self_att_block = NormalAndShiftWinsBlockPair(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_dual_path=use_dual_path,
            use_cross_attr=False,  # note here.
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_activation_func=mlp_activation_func,
            mlp_drop_ratio=mlp_drop_ratio,
        )
        # cross-attention block
        self.cross_att_block = NormalAndShiftWinsBlockPair(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_dual_path=use_dual_path,
            use_cross_attr=True,  # note here.
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_activation_func=mlp_activation_func,
            mlp_drop_ratio=mlp_drop_ratio,
        )

    def forward(self, x, y=None):
        if self.use_dual_path:
            x, y = self.self_att_block(x=x, y=y)
            x, y = self.cross_att_block(x=x, y=y)
            return x, y
        else:
            x = self.self_att_block(x=x, y=None)
            x = self.cross_att_block(x=x, y=None)
            return x

    def forward_(self, x, y):
        return self(x, y)


def test_self_and_cross_att_block():
    block = SelfAndCrossBlockPair(
        in_out_dims=2,
        num_heads=8,
        dims_per_head=2,
        window_size=(7, 7),
        use_dual_path=False,
        use_qkv_bias=True,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_hidden_dims=2 * 4,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
    )
    x = torch.rand(size=(20, 2, 224, 224))

    block(x)
    pass


if __name__ == '__main__':
    test_self_and_cross_att_block()

