from torch import nn
from a005_Block import Block


class NormalAndShiftWinsBlockPair(nn.Module):
    """
    NormalAndShiftWinsBlockPair class.

    Args:
        in_out_dims (int): input/output dimensions.
        num_heads (int): number of attention heads.
        dims_per_head (int): dimensions per attention head.
        window_size (tuple): window size.
        use_dual_path (bool): whether to use dual path.
        use_cross_attr (bool): whether to use cross-attribute attention.
        use_qkv_bias (bool): whether to use QKV bias.
        attention_drop_ratio (float): attention dropout ratio.
        linear_after_att_drop_ratio (float): linear dropout ratio after attention.
        mlp_hidden_dims (int): hidden dimensions of MLP.
        mlp_activation_func (nn.Module): activation function of MLP.
        mlp_drop_ratio (float): MLP dropout ratio.

    """

    def __init__(
        self,
        # the following params are copied from Block class __init__().
        # except that use_cyclic_shift is removed because it is determined here, not outside
        in_out_dims: int,
        num_heads: int,
        dims_per_head: int,
        window_size: tuple,
        use_dual_path: bool,
        use_cross_attr: bool,
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
        self.use_cross_attr = use_cross_attr
        self.use_qkv_bias = use_qkv_bias
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation_func = mlp_activation_func
        self.mlp_drop_ratio = mlp_drop_ratio

        # block_1 does not shift window
        self.normal_window_block = Block(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_cyclic_shift=False,  # note here.
            use_dual_path=use_dual_path,
            use_cross_attr=use_cross_attr,
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_activation_func=mlp_activation_func,
            mlp_drop_ratio=mlp_drop_ratio,
        )

        # block_2 does shift window
        self.shifted_window_block = Block(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_cyclic_shift=True,  # note here.
            use_dual_path=use_dual_path,
            use_cross_attr=use_cross_attr,
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_activation_func=mlp_activation_func,
            mlp_drop_ratio=mlp_drop_ratio,
        )

    def forward(self, x, y):
        """
        Forward pass.

        Args:
            x (torch.Tensor): input tensor.
            y (torch.Tensor | None)

        Returns:
            1 or 2 tensors

        """
        if self.use_dual_path:
            x, y = self.normal_window_block.forward(x=x, y=y)
            x, y = self.shifted_window_block.forward(x=x, y=y)
            return x, y
        else:
            x = self.normal_window_block.forward(x=x, y=None)
            x = self.shifted_window_block.forward(x=x, y=None)
            return x
