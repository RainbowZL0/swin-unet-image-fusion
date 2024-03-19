from torch import nn, Tensor
from a001_WindowAttention import WindowAttention


class AutoPathWinAtt(nn.Module):
    def __init__(
        self,
        # the following are copied from WindowAttention __init__()
        in_out_dims: int,
        num_heads: int,
        dims_per_head: int,
        window_size: tuple,
        use_cyclic_shift: bool,
        use_dual_path: bool,
        use_cross_att: bool,
        use_qkv_bias: bool,
        attention_drop_ratio: float,
        linear_after_att_drop_ratio: float,
    ):
        super().__init__()
        """copy init arguments above to attributes"""
        self.in_out_dims = in_out_dims
        self.num_heads = num_heads
        self.dims_per_head = dims_per_head
        self.window_size = window_size
        self.use_cyclic_shift = use_cyclic_shift
        self.use_dual_path = use_dual_path
        self.use_cross_att = use_cross_att
        self.use_qkv_bias = use_qkv_bias
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio

        self.window_attention_x = WindowAttention(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_cyclic_shift=use_cyclic_shift,
            use_cross_attention=use_cross_att,
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
        )
        if self.use_dual_path:
            """虽然一样的构造参数，但是不建议用copy.deepcopy，因为随机初始化状态会变得一样，这不是好的"""
            self.window_attention_y = WindowAttention(
                in_out_dims=in_out_dims,
                num_heads=num_heads,
                dims_per_head=dims_per_head,
                window_size=window_size,
                use_cyclic_shift=use_cyclic_shift,
                use_cross_attention=use_cross_att,
                use_qkv_bias=use_qkv_bias,
                attention_drop_ratio=attention_drop_ratio,
                linear_after_att_drop_ratio=linear_after_att_drop_ratio,
            )

    def forward(self, x, y):
        """
        Args:
            x (Tensor): 4D, (batch_size, in_dims, h, w)
            y : same shape as x if given

        Returns:
            1 or 2 tensors of shape 4D, (batch_size, out_dims, h, w)
        """
        if self.use_dual_path:
            if self.use_cross_att:
                # 选cross，则交叉qkv
                x, y = (
                    self.window_attention_x.forward(q=x, k=y, v=y),
                    self.window_attention_y.forward(q=y, k=x, v=x),
                )
            else:
                # 不选cross，则两路互不干扰
                x, y = (
                    self.window_attention_x.forward(q=x, k=x, v=x),
                    self.window_attention_y.forward(q=y, k=y, v=y)
                )
            return x, y
        else:
            return self.window_attention_x.forward(q=x, k=x, v=x)
