from collections import deque

import torch
from torch import nn
from tqdm import tqdm

from a010_StateRecorder import StateRecorder
from a012_SelfAndCrossBlockPair import SelfAndCrossBlockPair
from a006_PaddingOperation import MyPadding
from a011_PatchOperation import PatchMergingAndLinearLayer

# class MyModel(nn.Module):
#     def __init__(
#         self,
#         in_dims: int,
#         out_dims: int,
#         patch_merging_size_recorder: StateRecorder,
#     ):


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
) -> list:
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
        window_size=window_size,
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

    m_list = [
        padding_1,
        merging,
        padding_2,
        four_blocks,
    ]

    if belongs_to_encoder:
        return m_list
    else:
        return m_list[::-1]


# 假设in_dims_list = [3, 24, 48, 96]
# out_dims_list = [24, 48, 96, 192]
def generate_model_using_deque(
    window_size: tuple,
    # feature_shape_recorder: StateRecorder,
    # padding_size_recorder: StateRecorder,
    merging_size: tuple,
    in_dims_list: list,  # note here
    out_dims_list: list,  # note here
    # patch_merging_size_recorder: StateRecorder,
    att_num_heads: int,
    # att_dims_per_head: int,
    attention_drop_ratio: float,
    linear_after_att_drop_ratio: float,
    # mlp_hidden_dims: int,
    mlp_activation_func: nn.Module,
    mlp_drop_ratio: float,
):
    feature_shape_recorder = StateRecorder()
    padding_size_recorder = StateRecorder()
    patch_merging_size_recorder = StateRecorder()

    encoder_que = deque()
    decoder_que = deque()

    shape_reduction_times = len(in_dims_list)
    for i in range(shape_reduction_times):
        j = len(in_dims_list) - i - 1
        encoder_que.appendleft(
            get_encoder_or_decoder_block(
                mode="encoder",
                window_size=window_size,
                feature_shape_recorder=feature_shape_recorder,
                padding_size_recorder=padding_size_recorder,
                merging_size=merging_size,
                in_dims=in_dims_list[j],
                out_dims=out_dims_list[j],
                patch_merging_size_recorder=patch_merging_size_recorder,
                att_num_heads=att_num_heads,
                att_dims_per_head=out_dims_list[j],
                attention_drop_ratio=attention_drop_ratio,
                linear_after_att_drop_ratio=0.0,
                mlp_hidden_dims=out_dims_list[j] * 2,
                mlp_activation_func=mlp_activation_func,
                mlp_drop_ratio=mlp_drop_ratio,
            )
        )
        decoder_que.append(
            get_encoder_or_decoder_block(
                mode="decoder",
                window_size=window_size,
                feature_shape_recorder=feature_shape_recorder,
                padding_size_recorder=padding_size_recorder,
                merging_size=merging_size,
                in_dims=out_dims_list[j],
                out_dims=in_dims_list[j],
                patch_merging_size_recorder=patch_merging_size_recorder,
                att_num_heads=att_num_heads,
                att_dims_per_head=in_dims_list[j],
                attention_drop_ratio=attention_drop_ratio,
                linear_after_att_drop_ratio=linear_after_att_drop_ratio,
                mlp_hidden_dims=in_dims_list[j] * 2,
                mlp_activation_func=mlp_activation_func,
                mlp_drop_ratio=mlp_drop_ratio,
            )
        )
    return encoder_que, decoder_que


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
            x, y = m.forward(x=x, y=y)


def test_generate_model_using_deque():
    encoder_que, decoder_que = generate_model_using_deque(
        window_size=(8, 8),
        merging_size=(2, 2),
        in_dims_list=[1, 24, 48, 96],
        out_dims_list=[24, 48, 96, 192],
        att_num_heads=8,
        attention_drop_ratio=0.0,
        linear_after_att_drop_ratio=0.0,
        mlp_activation_func=nn.ELU(),
        mlp_drop_ratio=0.0,
    )

    a = torch.randn(size=(2, 1, 256, 256))
    b = torch.randn(size=(2, 1, 256, 256))

    result_recorder = StateRecorder()
    x, y = a, b
    for i, m_list in enumerate(encoder_que):
        for m in m_list:
            x, y = m.forward(x=x, y=y)
        if i < len(encoder_que) - 1:
            result_recorder.record((x, y))
    for j, m_list in enumerate(decoder_que):
        if j > 0:
            history_x, history_y = result_recorder.read()
            x += history_x
            y += history_y
        for m in m_list:
            x, y = m.forward(x=x, y=y)
    print(x.shape)


if __name__ == "__main__":
    test_generate_model_using_deque()
