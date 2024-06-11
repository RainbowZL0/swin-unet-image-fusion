import math
from collections import deque

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from a006_PaddingOperation import MyPadding
from a008_loss import MyLoss
from a010_StateRecorder import StateRecorder
from a011_PatchOperation import PatchMergingAndLinearLayer
from a012_SelfAndCrossBlockPair import SelfAndCrossBlockPair
from a009_NormalAndShiftWinsBlockPair import NormalAndShiftWinsBlockPair


class MyModel(nn.Module):
    def __init__(
            self,
            window_size: tuple,
            # feature_shape_recorder: StateRecorder,  # 在类内部生成Recorder，而不是构造时传入
            # padding_size_recorder: StateRecorder,
            merging_size: tuple,
            in_dims_list: list,  # note here
            out_dims_list: list,  # note here
            # patch_merging_size_recorder: StateRecorder,
            att_num_heads: int,
            att_dims_per_head_ratio: float,
            attention_drop_ratio: float,
            linear_after_att_drop_ratio: float,
            mlp_hidden_dims_ratio: int,
            mlp_activation_func: nn.Module,
            mlp_drop_ratio: float,
            # 最后一层相关
            final_layer_att_dims_per_head_ratio: float,
            final_conv_layer_kernel_size: int,
            final_layer_mlp_hidden_dims_ratio: int,
    ):
        super().__init__()
        self.window_size = window_size
        self.merging_size = merging_size
        self.in_dims_list = in_dims_list
        self.out_dims_list = out_dims_list
        self.att_num_heads = att_num_heads
        self.att_dims_per_head_ratio = att_dims_per_head_ratio
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio
        self.mlp_hidden_dims_ratio = mlp_hidden_dims_ratio
        self.mlp_activation_func = mlp_activation_func
        self.mlp_drop_ratio = mlp_drop_ratio
        self.final_layer_att_dims_per_head_ratio = final_layer_att_dims_per_head_ratio
        self.final_layer_conv_kernel_size = final_conv_layer_kernel_size
        self.final_layer_mlp_hidden_dims_ratio = final_layer_mlp_hidden_dims_ratio

        # 创建三个recorder
        self.feature_shape_recorder = StateRecorder()
        self.padding_size_recorder = StateRecorder()
        self.patch_merging_size_recorder = StateRecorder()

        # 为了在运算中记录u-net的中间计算结果，还需要一个recorder，每次forward前清空，然后在forward中使用
        self.u_net_intermediate_result_recorder = StateRecorder()

        # 创建model，分为encoder和decoder两部分
        self.encoder_list, self.decoder_list = self.generate_model_using_deque()

        """实验最后一层"""
        # 最后的合并通道层，输出最终结果
        self.final_layer = self.get_final_layer()

        # # prepare stage
        # self.prepare_stage = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(4),
        #     mlp_activation_func,
        #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     mlp_activation_func,
        #
        # )
        # self.final_merge_channel_layer = nn.Conv2d(
        #     in_channels=2,
        #     out_channels=1,
        #     kernel_size=MyConfig.FINAL_MERGING_LAYER_KERNEL_SIZE,
        #     padding="same",
        # )

        # # prepare stage
        # self.prepare_stage = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(4),
        #     mlp_activation_func,
        #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(8),
        #     mlp_activation_func,
        #
        # )

    def get_final_layer(self):
        # final_att = SelfAndCrossBlockPair(
        #     in_out_dims=2,
        #     num_heads=self.att_num_heads,
        #     dims_per_head=math.floor(2 * self.final_layer_att_dims_per_head_ratio),
        #     window_size=self.window_size,
        #     use_dual_path=False,
        #     use_qkv_bias=True,
        #     attention_drop_ratio=self.attention_drop_ratio,
        #     linear_after_att_drop_ratio=self.linear_after_att_drop_ratio,
        #     mlp_hidden_dims=2 * self.final_layer_mlp_hidden_dims_ratio,
        #     mlp_activation_func=self.mlp_activation_func,
        #     mlp_drop_ratio=self.mlp_drop_ratio,
        # )
        # final_att = NormalAndShiftWinsBlockPair(
        #     in_out_dims=2,
        #     num_heads=self.att_num_heads,
        #     dims_per_head=math.floor(2 * self.final_layer_att_dims_per_head_ratio),
        #     window_size=self.window_size,
        #     use_dual_path=False,
        #     use_cross_attr=False,
        #     use_qkv_bias=True,
        #     attention_drop_ratio=self.attention_drop_ratio,
        #     linear_after_att_drop_ratio=self.linear_after_att_drop_ratio,
        #     mlp_hidden_dims=2 * self.final_layer_mlp_hidden_dims_ratio,
        #     mlp_activation_func=self.mlp_activation_func,
        #     mlp_drop_ratio=self.mlp_drop_ratio,
        # )
        final_conv_1 = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=self.final_layer_conv_kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        final_norm = nn.BatchNorm2d(2)
        final_activation = self.mlp_activation_func
        final_conv_2 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=self.final_layer_conv_kernel_size,
            padding="same",
            padding_mode="reflect",
        )
        return nn.Sequential(
            # final_att,
            final_conv_1,
            final_norm,
            final_activation,
            final_conv_2
        )

    def do_final_layer(self, x, y):
        feature = torch.concat(tensors=[x, y], dim=1)
        return self.final_layer(feature)

    def generate_model_using_deque(self):
        """
        Returns: ModuleList
        """
        encoder_que = deque()
        decoder_que = deque()

        shape_reduction_times = len(self.in_dims_list)
        for j in range(shape_reduction_times - 1, -1, -1):
            encoder_que.appendleft(
                get_encoder_or_decoder_block(
                    mode="encoder",
                    window_size=self.window_size,
                    feature_shape_recorder=self.feature_shape_recorder,
                    padding_size_recorder=self.padding_size_recorder,
                    merging_size=self.merging_size,
                    in_dims=self.in_dims_list[j],
                    out_dims=self.out_dims_list[j],
                    patch_merging_size_recorder=self.patch_merging_size_recorder,
                    att_num_heads=self.att_num_heads,
                    att_dims_per_head=math.floor(self.out_dims_list[j] * self.att_dims_per_head_ratio),
                    attention_drop_ratio=self.attention_drop_ratio,
                    linear_after_att_drop_ratio=self.linear_after_att_drop_ratio,
                    mlp_hidden_dims=self.out_dims_list[j] * self.mlp_hidden_dims_ratio,
                    mlp_activation_func=self.mlp_activation_func,
                    mlp_drop_ratio=self.mlp_drop_ratio,
                )
            )
            decoder_que.append(
                get_encoder_or_decoder_block(
                    mode="decoder",
                    window_size=self.window_size,
                    feature_shape_recorder=self.feature_shape_recorder,
                    padding_size_recorder=self.padding_size_recorder,
                    merging_size=self.merging_size,
                    in_dims=self.out_dims_list[j],
                    out_dims=self.in_dims_list[j],
                    patch_merging_size_recorder=self.patch_merging_size_recorder,
                    att_num_heads=self.att_num_heads,
                    att_dims_per_head=math.floor(self.out_dims_list[j] * self.att_dims_per_head_ratio),
                    attention_drop_ratio=self.attention_drop_ratio,
                    linear_after_att_drop_ratio=self.linear_after_att_drop_ratio,
                    mlp_hidden_dims=self.in_dims_list[j] * self.mlp_hidden_dims_ratio,
                    mlp_activation_func=self.mlp_activation_func,
                    mlp_drop_ratio=self.mlp_drop_ratio,
                )
            )

        # for elem in encoder_que:
        #     print(type(elem))

        # 包装为ModuleList类，然后返回
        encoder_module_list, decoder_module_list = nn.ModuleList(encoder_que), nn.ModuleList(decoder_que)
        return encoder_module_list, decoder_module_list

    def forward(self, in_x, in_y) -> torch.Tensor:
        # 为了保险起见，每次forward前清空recorder，但是正常情况下，上次forward结束后，栈里应该已经是空的
        self.u_net_intermediate_result_recorder.delete_all()

        # u-net部分forward
        x, y = in_x, in_y
        for i, m_list in enumerate(self.encoder_list):
            m_list: nn.ModuleList
            for m in m_list:
                x, y = m(x=x, y=y)
            if i < len(self.encoder_list) - 1:
                self.u_net_intermediate_result_recorder.record((x, y))
        for j, m_list in enumerate(self.decoder_list):
            if j > 0:
                history_x, history_y = self.u_net_intermediate_result_recorder.read()
                x += history_x
                y += history_y
            for m in m_list:
                x, y = m(x=x, y=y)

        # 合并x, y
        return self.do_final_layer(x, y)

    def forward_(self, in_x, in_y) -> torch.Tensor:
        return self(in_x, in_y)


def get_encoder_or_decoder_block(
        mode: str,
        window_size: tuple,
        feature_shape_recorder: StateRecorder,
        padding_size_recorder: StateRecorder,
        merging_size: tuple,
        in_dims: int,
        out_dims: int,
        patch_merging_size_recorder: StateRecorder,
        att_num_heads: int,
        att_dims_per_head: int,
        attention_drop_ratio: float,
        linear_after_att_drop_ratio: float,
        mlp_hidden_dims: int,
        mlp_activation_func: nn.Module,
        mlp_drop_ratio: float,
) -> nn.ModuleList:
    if mode == "encoder":
        belongs_to_encoder = True
        attr_in_out_dims = out_dims
    elif mode == "decoder":
        belongs_to_encoder = False
        attr_in_out_dims = in_dims
    else:
        raise ValueError("mode must be either encoder or decoder")

    padding_1 = MyPadding(
        belongs_to_encoder=belongs_to_encoder,
        window_size=merging_size,
        use_dual_path=True,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )

    merging = PatchMergingAndLinearLayer(
        belongs_to_encoder=belongs_to_encoder,
        use_dual_path=True,
        merging_or_unmerging_size=merging_size,
        in_dims=in_dims,
        out_dims=out_dims,
        patch_merging_size_recorder=patch_merging_size_recorder,
        activation_func=mlp_activation_func,
    )

    padding_2 = MyPadding(
        belongs_to_encoder=belongs_to_encoder,
        window_size=window_size,
        use_dual_path=True,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )

    four_blocks = SelfAndCrossBlockPair(
        in_out_dims=attr_in_out_dims,
        num_heads=att_num_heads,
        dims_per_head=att_dims_per_head,
        window_size=window_size,
        use_dual_path=True,
        use_qkv_bias=True,
        attention_drop_ratio=attention_drop_ratio,
        linear_after_att_drop_ratio=linear_after_att_drop_ratio,
        mlp_hidden_dims=mlp_hidden_dims,
        mlp_activation_func=mlp_activation_func,
        mlp_drop_ratio=mlp_drop_ratio,
    )

    m_list = nn.ModuleList(
        [
            padding_1,
            merging,
            padding_2,
            four_blocks,
        ]
    )

    if belongs_to_encoder:
        return m_list
    else:
        return m_list[::-1]


def test_u_shape():
    feature_shape_recorder = StateRecorder()
    padding_size_recorder = StateRecorder()
    patch_merging_size_recorder = StateRecorder()

    patch_merging = PatchMergingAndLinearLayer(
        belongs_to_encoder=True,
        use_dual_path=True,
        merging_or_unmerging_size=(2, 2),
        in_dims=3,
        out_dims=6,
        patch_merging_size_recorder=patch_merging_size_recorder,
    )

    padding = MyPadding(
        belongs_to_encoder=True,
        window_size=(7, 7),
        use_dual_path=True,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )

    self_and_cross_block_pair = SelfAndCrossBlockPair(
        in_out_dims=6,
        num_heads=8,
        dims_per_head=8,
        window_size=(7, 7),
        use_dual_path=True,
        use_qkv_bias=True,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_hidden_dims=12,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
    )

    self_and_cross_block_pair_2 = SelfAndCrossBlockPair(
        in_out_dims=6,
        num_heads=8,
        dims_per_head=8,
        window_size=(7, 7),
        use_dual_path=True,
        use_qkv_bias=True,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_hidden_dims=12,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
    )

    padding_2 = MyPadding(
        belongs_to_encoder=False,
        window_size=(7, 7),
        use_dual_path=True,
        feature_shape_recorder=feature_shape_recorder,
        padding_size_recorder=padding_size_recorder,
    )

    patch_merging_2 = PatchMergingAndLinearLayer(
        belongs_to_encoder=False,
        use_dual_path=True,
        merging_or_unmerging_size=tuple(),
        in_dims=6,
        out_dims=3,
        patch_merging_size_recorder=patch_merging_size_recorder,
    )

    m_list = nn.ModuleList(
        [
            patch_merging,
            padding,
            self_and_cross_block_pair,
            self_and_cross_block_pair_2,
            padding_2,
            patch_merging_2,
        ]
    )

    for _ in tqdm(range(1000)):
        a = torch.rand(size=(1, 3, 96, 96))
        b = a.clone()
        x, y = a, b
        for m in m_list:
            x, y = m(x=x, y=y)


# 假设in_dims_list = [3, 24, 48, 96]
# out_dims_list = [24, 48, 96, 192]


def test_generate_model_using_deque():
    device = torch.device("cpu")
    my_model = MyModel(
        window_size=(7, 7),
        merging_size=(2, 2),
        in_dims_list=[1, 24, 48, 96, 192],
        out_dims_list=[24, 48, 96, 192, 384],
        att_num_heads=8,
        att_dims_per_head_ratio=1 / 8,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_hidden_dims_ratio=4,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
        final_layer_att_dims_per_head_ratio=1,
        final_conv_layer_kernel_size=3,
        final_layer_mlp_hidden_dims_ratio=1,
    ).to(device=device)
    my_loss = MyLoss().to(device=device)

    a = torch.randn(size=(2, 1, 200, 200)).to(device=device)  # b, c, h, w
    b = torch.randn(size=(2, 1, 200, 200)).to(device=device)

    optimizer = Adam(
        params=my_model.parameters(),
        lr=1e-3,
    )

    for _ in tqdm(range(1000)):
        fusion = my_model(in_x=a, in_y=b)

        loss, _ = my_loss.calcu_total_loss(
            fusion_images=fusion,
            ir_images=a,
            vis_images=b,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(fusion.shape)
        # print(my_loss.loss_recorder.peek())


if __name__ == "__main__":
    test_generate_model_using_deque()
