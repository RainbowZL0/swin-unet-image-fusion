from typing import Callable

from torch import Tensor, nn

from a007_utils import *
from a010_StateRecorder import StateRecorder
from a004_AddAndLayerNormWithOtherModule import my_layer_norm


class PatchMergingAndLinearLayer(nn.Module):
    """
    This class implements a patch merging layer.

    Args:
        belongs_to_encoder (bool): Whether the layer is used in the encoder or decoder.
        use_dual_path (bool): Whether to use the dual path or not.
        patch_merging_size_recorder (StateRecorder): A StateRecorder object to record the patch merging size.
        merging_or_unmerging_size (tuple): A tuple of integers,
            describing the size of the patch merging or unmerging operation.
            When encoder, must give a tuple value. When decoder, pass an empty tuple or use the default one.
            Decoder merging_size will read automatically from recorder.

    """

    def __init__(
        self,
        belongs_to_encoder: bool,
        use_dual_path: bool,
        in_dims: int,
        out_dims: int,
        patch_merging_size_recorder: StateRecorder,
        merging_or_unmerging_size: tuple,
        activation_func: nn.Module = nn.ELU(),
    ):
        super().__init__()
        """copy init arguments to attributes"""
        # encoder用do_merging, decoder用undo_merging
        self.belongs_to_encoder = belongs_to_encoder
        # 决定输入1个或2个向量
        self.use_dual_path = use_dual_path
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.patch_merging_recorder = patch_merging_size_recorder
        # (h, w), describe how many points will be merged to a single point
        self.merging_or_unmerging_size = merging_or_unmerging_size
        self.activation_func = activation_func

        """新参数，计算linear的输入输出通道数量 把conv当做linear用"""
        h, w = merging_or_unmerging_size
        channel_changed_ratio = h * w
        if belongs_to_encoder:
            self.conv_in_dims = in_dims * channel_changed_ratio
            self.conv_out_dims = out_dims
        else:
            self.conv_in_dims = in_dims
            self.conv_out_dims = out_dims * channel_changed_ratio

        """conv部分"""
        # self.mlp_is_initialized = False
        self.mlp_layer_x = nn.Conv2d(in_channels=self.conv_in_dims, out_channels=self.conv_out_dims, kernel_size=1)
        self.layer_norm_x = nn.LayerNorm(normalized_shape=self.conv_out_dims)
        if self.use_dual_path:
            self.mlp_layer_y = nn.Conv2d(in_channels=self.conv_in_dims, out_channels=self.conv_out_dims, kernel_size=1)
            self.layer_norm_y = nn.LayerNorm(normalized_shape=self.conv_out_dims)

        """a buffer to show device where the class object is"""
        self.register_buffer(name="buffer_to_show_device", tensor=torch.zeros(size=(1,)))

        """根据encoder或decoder决定操作顺序"""
        self.func_list = self.build_func_list()

    def do_patch_merging_for_one_tensor(self, feature: Tensor) -> Tensor:
        """
        This function performs the patch merging operation.

        Args:
            feature (Tensor): The input feature tensor, with shape (batch_size, channel, height, width).

        Returns:
            Tensor: The output feature tensor, with shape (batch_size, h_new, w_new, :).

        """
        n_points_h, n_points_w = self.merging_or_unmerging_size

        # now feature_map shape is (batch_size, channel, h_new, w_new, n_points_h, n_points_w)
        # then concat pixels from each patch, on channel's dim
        feature = einops.rearrange(
            tensor=feature,
            pattern="b c (n_wins_h n_points_h) (n_wins_w n_points_w) ->"
                    "b (n_points_h n_points_w c) n_wins_h n_wins_w",
            n_points_h=n_points_h,
            n_points_w=n_points_w,
        )
        return feature

    def undo_patch_merging_for_one_tensor(self, feature: Tensor) -> Tensor:
        """
        This function performs the patch unmerging operation.

        Args:
            feature (Tensor): The input feature tensor, with shape (batch_size, channel, h_new, w_new, :).

        Returns:
            Tensor: The output feature tensor, with shape (batch_size, channel, height, width).

        """
        if self.merging_or_unmerging_size == (1, 1):
            return feature
        else:
            undo_merging_size_h, undo_merging_size_w = self.merging_or_unmerging_size
            return einops.rearrange(
                tensor=feature,
                pattern="b (n_points_h_per_win n_points_w_per_win channels) n_wins_h n_wins_w ->"
                " b channels (n_wins_h n_points_h_per_win) (n_wins_w n_points_w_per_win)",
                n_points_h_per_win=undo_merging_size_h,
                n_points_w_per_win=undo_merging_size_w,
            )

    def undo_patch_merging_for_one_tensor_2(self, feature: Tensor) -> Tensor:
        """
        This function performs the patch unmerging operation, using only permute and reshape operations.
        It is equivalent to the undo_patch_merging function.

        Args:
            feature (Tensor): The input feature tensor, with shape (batch_size, channel, h_new, w_new, :).

        Returns:
            Tensor: The output feature tensor, with shape (batch_size, channel, height, width).

        """
        if self.merging_or_unmerging_size == (1, 1):
            return feature
        else:
            n_points_h_per_win, n_points_w_per_win = self.merging_or_unmerging_size
            batch_size, dims, n_wins_h, n_wins_w = feature.shape
            feature = feature.reshape(
                batch_size, n_points_h_per_win, n_points_w_per_win, -1, n_wins_h, n_wins_w
            )
            # () -> (batch_size, channels, n_wins_h, n_points_h_per_win, n_wins_w, n_points_w_per_win)
            feature = feature.permute(0, 3, 4, 1, 5, 2)
            # () -> (batch_size, channels, n_wins_h * n_points_h_per_win, n_wins_w * n_points_w_per_win)
            feature = feature.reshape(
                batch_size, -1, n_wins_h * n_points_h_per_win, n_wins_w * n_points_w_per_win
            )
            return feature

    def merge_or_unmerge_for_one_tensor(self, x):
        if self.belongs_to_encoder:
            return self.do_patch_merging_for_one_tensor(feature=x)
        else:
            return self.undo_patch_merging_for_one_tensor(feature=x)

    # def calcu_linear_in_or_out_dims(self):
    #     """
    #     Calculates the input dimensions of the linear layer.
    #
    #     Returns:
    #         int: The input dimensions of the linear layer.
    #
    #     """
    #     n_points_per_win_h, n_points_per_win_w = self.merging_or_unmerging_size
    #     n_points_per_win = n_points_per_win_h * n_points_per_win_w
    #     if self.belongs_to_encoder:
    #         # encoder计算linear_in_dims
    #         return self.in_dims * n_points_per_win
    #     else:
    #         # decoder计算linear_out_dims
    #         return self.out_dims * n_points_per_win

    # def record_merging_size(self):
    #     self.patch_merging_recorder.record(self.merging_or_unmerging_size)
    #
    # def read_merging_size_from_recorder(self):
    #     self.merging_or_unmerging_size = self.patch_merging_recorder.read()

    # def check_and_init_linear_layer(self):
    #     if not self.mlp_is_initialized:
    #         if self.belongs_to_encoder:
    #             in_dims = self.calcu_linear_in_or_out_dims()
    #             self.mlp_layer_x = nn.Conv2d(
    #                 in_channels=in_dims,
    #                 out_channels=self.out_dims,
    #                 kernel_size=1,
    #                 device=self.buffer_to_show_device.device
    #             )
    #             if self.use_dual_path:
    #                 self.mlp_layer_y = nn.Conv2d(
    #                     in_channels=in_dims,
    #                     out_channels=self.out_dims,
    #                     kernel_size=1,
    #                     device=self.buffer_to_show_device.device
    #                 )
    #         else:
    #             out_dims = self.calcu_linear_in_or_out_dims()
    #             self.mlp_layer_x = nn.Conv2d(
    #                 in_channels=self.in_dims,
    #                 out_channels=out_dims,
    #                 kernel_size=1,
    #                 device=self.buffer_to_show_device.device
    #             )
    #             if self.use_dual_path:
    #                 self.mlp_layer_y = nn.Conv2d(
    #                     in_channels=self.in_dims,
    #                     out_channels=out_dims,
    #                     kernel_size=1,
    #                     device=self.buffer_to_show_device.device
    #                 )
    #         # 初始化后，修改初始化状态为True
    #         self.mlp_is_initialized = True

    def merge_or_unmerge(self, x, y=None):
        if y is not None:
            x, y = (self.merge_or_unmerge_for_one_tensor(elem) for elem in (x, y))
            return x, y
        else:
            return self.merge_or_unmerge_for_one_tensor(x), None

    def do_linear(self, x, y=None):
        if y is not None:
            return self.mlp_layer_x(x), self.mlp_layer_y(y)
        else:
            return self.mlp_layer_x(x), None

    def do_layer_norm(self, x, y=None):
        if y is not None:
            return my_layer_norm(x=x, layer_x=self.layer_norm_x, y=y, layer_y=self.layer_norm_y)
        else:
            return my_layer_norm(x=x, layer_x=self.layer_norm_x), None

    def do_activation(self, x, y=None):
        if y is not None:
            return self.activation_func(x), self.activation_func(y)
        else:
            return self.activation_func(x), None

    def build_func_list(self):
        func_list = [self.merge_or_unmerge, self.do_linear, self.do_layer_norm, self.do_activation]
        if self.belongs_to_encoder:
            return func_list
        else:
            func_list = [func_list[1], func_list[2], func_list[0], func_list[3]]
            return func_list

    def forward(self, x, y=None):
        """做patch_merging相关操作"""

        # """处理merging_size问题"""
        # if self.belongs_to_encoder:
        #     self.record_merging_size()
        # else:
        #     self.read_merging_size_from_recorder()
        #
        # """设定好merging_size后，若没有初始化linear_layer，初始化。"""
        # self.check_and_init_linear_layer()

        """操作"""
        for func in self.func_list:
            func: Callable
            x, y = func(x, y)

        if y is not None:
            return x, y
        else:
            return x

    def forward_(self, x, y):
        return self(x, y)
