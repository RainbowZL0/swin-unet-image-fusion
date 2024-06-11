from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional

from a010_StateRecorder import StateRecorder


class MyPadding(nn.Module):
    def __init__(
        self,
        belongs_to_encoder: bool,
        window_size: tuple,
        use_dual_path: bool,
        feature_shape_recorder: StateRecorder,
        padding_size_recorder: StateRecorder,
    ):
        super().__init__()
        """copy arguments to attributes"""
        # 如果是encoder的一部分，则做padding，否则是decoder的一部分，做反向padding
        self.belongs_to_encoder = belongs_to_encoder
        self.window_size = window_size
        self.use_dual_path = use_dual_path

        """new attributes"""
        # 初始化输入特征图形状为空tuple。首次forward时记录。
        # 如果属于encoder，则表示padding之前的shape。如果属于decoder，则表示反向padding之后的shape。
        self.feature_shape_hw: tuple = tuple()
        # 下边和右边padding的size, (down, right)
        # 如果属于encoder，则取值为正数或0。如果属于decoder，则为需要反向padding的size。
        self.padding_size: tuple = tuple()

        """state recorder"""
        self.feature_shape_recorder = feature_shape_recorder
        self.padding_size_recorder = padding_size_recorder

    def initialize_and_record_feature_shape_hw_if_have_not(self, x):
        """q shape is (batch_size, channels, height, width)"""
        _, _, h, w = x.shape
        # 训练时只更新一次
        if self.training:
            if (
                not self.feature_shape_hw
            ):  # 空tuple判定为False，用not转为True。所以空tuple时进入if
                self.feature_shape_hw = (h, w)
        # 推理时次次更新
        else:
            self.feature_shape_hw = (h, w)

        # 训练和推理时都次次记录
        self.feature_shape_recorder.record(self.feature_shape_hw)

    @staticmethod
    def calculate_padding_size(current_length, window_size):
        return (window_size - current_length % window_size) % window_size

    def get_feature_shape_hw(self):
        return self.feature_shape_hw

    def set_feature_shape_hw(self, new_feature_shape_hw):
        """如果需要pad，可以选择更新已经记录的属性feature_shape_hw为pad之后的。推荐不更新。
        另一个用法是，当MyPadding对象用于decoder，需要从外部直接设定feature_shape_hw的值，通过本函数接口。
        """
        self.feature_shape_hw = new_feature_shape_hw

    def initialize_and_record_padding_size_for_window_partition_if_have_not(self):
        """计算需要padding多少。并记录。"""
        feature_shape_h, feature_shape_w = self.feature_shape_hw
        win_size_h, win_size_w = self.window_size
        if self.training:
            if not self.padding_size:  # if this tuple is empty, initialize
                self.padding_size = (
                    self.calculate_padding_size(feature_shape_h, win_size_h),
                    self.calculate_padding_size(feature_shape_w, win_size_w),
                )
        else:
            self.padding_size = (
                self.calculate_padding_size(feature_shape_h, win_size_h),
                self.calculate_padding_size(feature_shape_w, win_size_w),
            )
        # 训练和推理时都次次记录
        self.padding_size_recorder.record(self.padding_size)

    def update_feature_shape_hw_after_padding(self):
        # 如果需要pad，更新记录的self.feature_shape_hw
        if self.padding_size[0] != 0 or self.padding_size[1] != 0:
            new_feature_shape_hw = tuple(
                i1 + i2 for i1, i2 in zip(self.feature_shape_hw, self.padding_size)
            )
            self.set_feature_shape_hw(new_feature_shape_hw=new_feature_shape_hw)

    def do_padding(
        self, x: Tensor, y: Optional[Tensor]
    ) -> tuple[Tensor, Optional[Tensor]]:
        """
        表层api，判断需要pad一个或两个tensor，然后pad。
        Args:
            x (Tensor)
            y (Tensor | None)

        Returns:
            tuple(Tensor, Tensor) or tuple(Tensor, None)
        """
        if y is not None:
            x, y = [self.do_padding_for_one_tensor(elem) for elem in [x, y]]
            return x, y
        else:
            return self.do_padding_for_one_tensor(x), None

    def do_padding_for_one_tensor(self, tensor) -> Tensor:
        """
        前提是self.padding_size已经被初始化。
        只pad一个tensor。是被封装的函数。
        Args:
            tensor (Tensor)

        Returns:
             Tensor:

        """
        padding_down, padding_right = self.padding_size
        # 如果高宽都不需要padding，直接return
        if padding_down == 0 and padding_right == 0:
            return tensor
        # 否则padding后return
        else:
            return functional.pad(
                # order of pad tuple is not the way you think, refer to documentation for details
                input=tensor, pad=(0, padding_right, 0, padding_down), mode="reflect"
            )

    def undo_padding_for_one_tensor(self, tensor) -> Tensor:
        """
        假设调用时self.padding_size已经获取到，代表反向pad。返回撤销pad之后的tensor。
        Args:
            tensor: shape 4D, (batch_size, channels, height, width)

        Returns:
            反向pad之后的tensor
        """
        # 这种写法兼容padding_h或padding_w等于0的情况，不用再做判断。
        tensor_h, tensor_w = tensor.shape[-2:]
        padding_h, padding_w = self.padding_size
        h_end, w_end = tensor_h - padding_h, tensor_w - padding_w
        return tensor[:, :, :h_end, :w_end]

    def undo_padding(self, x, y):
        if y is not None:
            x, y = [self.undo_padding_for_one_tensor(elem) for elem in [x, y]]
            return x, y
        else:
            return self.undo_padding_for_one_tensor(x), None

    def get_padding_size(self):
        return self.padding_size

    def set_padding_size(self, new_padding_size):
        self.padding_size = new_padding_size

    def read_feature_shape_hw_from_recorder(self):
        self.feature_shape_hw = self.feature_shape_recorder.read()

    def read_padding_size_from_recorder(self):
        self.padding_size = self.padding_size_recorder.read()

    def forward(self, x, y):
        if self.belongs_to_encoder:
            """encoder时需要初始化feature_shape和padding_size"""
            self.initialize_and_record_feature_shape_hw_if_have_not(x=x)
            self.initialize_and_record_padding_size_for_window_partition_if_have_not()
            """做padding"""
            if self.use_dual_path:
                return self.do_padding(x=x, y=y)
            else:
                x, _ = self.do_padding(x=x, y=None)
                return x
        else:
            """decoder时，从recorder读取feature_shape_hw和padding_size，不使用初始化方法"""
            self.read_feature_shape_hw_from_recorder()
            self.read_padding_size_from_recorder()
            """做反向padding"""
            if self.use_dual_path:
                return self.undo_padding(x=x, y=y)
            else:
                x, _ = self.undo_padding(x=x, y=None)
                return x

    def forward_(self, x, y):
        return self(x, y)


def test_stack():
    feature_shape_recorder = StateRecorder()
    padding_size_recorder = StateRecorder()

    padding = MyPadding(
        belongs_to_encoder=True,
        window_size=(7, 7),
        use_dual_path=False,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )
    a = torch.randn(size=(5, 3, 6, 6))
    a = padding(x=a, y=None)

    print(padding.feature_shape_recorder.record_stack)

    decoder_padding = MyPadding(
        belongs_to_encoder=False,
        window_size=(7, 7),
        use_dual_path=False,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )
    a = decoder_padding(x=a, y=None)
    print(padding.feature_shape_recorder.record_stack)


if __name__ == "__main__":
    test_stack()
