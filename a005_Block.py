import torch
from torch import nn, Tensor
from tqdm import tqdm

from a002_AutoPathWinAtt import AutoPathWinAtt
from a003_AutoPathMLP import AutoPathMLP
from a004_AddAndLayerNormWithOtherModule import AddAndLayerNormWithOtherModule


class Block(nn.Module):
    def __init__(
        self,
        # the following are copied from class AutoPathWinAtt __init__()
        in_out_dims: int,
        num_heads: int,
        dims_per_head: int,
        window_size: tuple,
        use_cyclic_shift: bool,
        use_dual_path: bool,
        use_cross_attr: bool,
        use_qkv_bias: bool,
        attention_drop_ratio: float,
        linear_after_att_drop_ratio: float,
        # from class AutoPathMLP __init__()
        mlp_hidden_dims: int,
        mlp_activation_func: nn.Module,
        mlp_drop_ratio: float,
    ):
        super().__init__()
        """copy init arguments above to attributes"""
        # following part is copied from class LNAndWindowAttention
        self.in_out_dims = in_out_dims
        self.num_heads = num_heads
        self.dims_per_head = dims_per_head
        self.window_size = window_size
        self.use_cyclic_shift = use_cyclic_shift
        self.use_dual_path = use_dual_path
        self.use_cross_attr = use_cross_attr
        self.use_qkv_bias = use_qkv_bias
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio
        # from class AutoPathMLP __init__()
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_activation_func = mlp_activation_func
        self.mlp_drop_ratio = mlp_drop_ratio

        """new attributes"""
        # forward时的输入和模型use_cross_attention的选项是否匹配 的状态记录。第一次forward时被初始化。先置为None。
        self.input_compatibility_with_cross_option = None

        self.auto_path_win_att = AutoPathWinAtt(
            in_out_dims=in_out_dims,
            num_heads=num_heads,
            dims_per_head=dims_per_head,
            window_size=window_size,
            use_cyclic_shift=use_cyclic_shift,
            use_dual_path=use_dual_path,
            use_cross_att=use_cross_attr,
            use_qkv_bias=use_qkv_bias,
            attention_drop_ratio=attention_drop_ratio,
            linear_after_att_drop_ratio=linear_after_att_drop_ratio,
        )

        self.auto_path_mlp = AutoPathMLP(
            in_out_dims=in_out_dims,
            hidden_dims=mlp_hidden_dims,
            activation_func=mlp_activation_func,
            use_dual_path=use_dual_path,
            drop_ratio=mlp_drop_ratio,
        )

        self.stage_1 = AddAndLayerNormWithOtherModule(
            normalized_shape=[in_out_dims],
            use_dual_path=use_dual_path,
            other_module=self.auto_path_win_att,
        )

        self.stage_2 = AddAndLayerNormWithOtherModule(
            normalized_shape=[in_out_dims],
            use_dual_path=use_dual_path,
            other_module=self.auto_path_mlp,
        )

    def check_compatibility_between_cross_and_path_option(self):
        """This could be done when instance is init, before the forward is really called."""
        if self.use_cross_attr:
            assert self.use_dual_path is True

    def check_input_compatibility_with_option(self, x, y):
        """If "use_cross_attention" is True, then y is not None,
        else y is None. Assure that calculation could continue.
        Args:
            x (Tensor)
            y (Tensor | None)

        """
        # 如果是首次调用forward，则需要检查，然后记录结果。后续再次调用forward时跳过检查。
        if self.input_compatibility_with_cross_option is None:
            assert (
                    x is not None
            ), "x should not be None"  # In any case, x must not be None.

            compatible = True
            # 如果选了dual_path，那么必须传入两个向量
            if self.use_dual_path and y is None:
                compatible = False
            elif not self.use_dual_path and y is not None:
                compatible = False

            # 如果选的是cross_attr，那么x和y不可能是每个元素完全相同的
            if self.use_cross_attr:
                if (x == y).all():
                    compatible = False

            if not compatible:
                self.input_compatibility_with_cross_option = False
                print("传入tensor情况和cross_attr, dual_path的选项不兼容。")
                exit()
            else:
                self.input_compatibility_with_cross_option = (
                    True  # 修改用于记录状态的属性
                )
        # 首次forward已经检查过，那么不再检查。
        else:
            return

    def forward(self, x, y):
        """
        Args:
            x (Tensor): 4D, (batch_size, in_dims, h, w)
            y : same shape as x if given, else None.

        Returns:
            Tensor: shape 4D, (batch_size, out_dims, h, w)
        """
        self.check_input_compatibility_with_option(x=x, y=y)

        if self.use_dual_path or y is not None:
            x, y = self.stage_1.forward(x, y)
            x, y = self.stage_2.forward(x, y)
            return x, y
        else:
            x = self.stage_1.forward(x=x, y=None)
            x = self.stage_2.forward(x=x, y=None)
            return x


def test_block():
    batch_size = 64
    channels = 40
    h, w = 49, 49

    device = "cpu"
    a = torch.randn(batch_size, channels, h, w).to(device=device)
    b = a.clone()
    block = Block(
        in_out_dims=channels,
        num_heads=4,
        dims_per_head=channels,
        window_size=(7, 7),
        use_cyclic_shift=True,
        use_dual_path=True,
        use_cross_attr=False,
        use_qkv_bias=True,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_hidden_dims=128,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
    ).to(device=device)

    for _ in tqdm(range(10000)):
        block.forward(x=a, y=b)


if __name__ == "__main__":
    test_block()
