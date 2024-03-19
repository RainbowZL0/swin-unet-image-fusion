import einops
import torch
import torch.nn.functional as f
from torch import Tensor, nn
from tqdm import tqdm


class WindowAttention(nn.Module):
    def __init__(
        self,
        in_out_dims: int,
        num_heads: int,
        dims_per_head: int,
        window_size: tuple,
        use_cyclic_shift: bool,
        use_cross_attention: bool,
        use_qkv_bias: bool,
        attention_drop_ratio: float,
        linear_after_att_drop_ratio: float,
    ):
        super().__init__()
        self.in_out_dims = in_out_dims
        self.num_heads = num_heads
        self.dims_per_head = dims_per_head
        self.window_size = window_size
        self.use_cyclic_shift = use_cyclic_shift
        self.use_cross_attention = use_cross_attention
        self.use_qkv_bias = use_qkv_bias
        self.attention_drop_ratio = attention_drop_ratio
        self.linear_after_att_drop_ratio = linear_after_att_drop_ratio

        self.qk_scale = (
            self.dims_per_head**-0.5
        )  # will be multiplied after the score is got.
        self.attention_drop_layer = nn.Dropout(self.attention_drop_ratio)
        self.linear_drop_layer = nn.Dropout(self.linear_after_att_drop_ratio)

        # 用于在forward时记录传入的特征图的feature_shape_hw，应该是一个元组，(h, w)。初始化为长度为0的tuple
        self.feature_shape_hw: tuple = tuple()

        # 所有heads先都叠在channels上。后续再分开成各自的head
        self.q_for_heads = nn.Linear(
            in_features=self.in_out_dims,
            out_features=self.num_heads * self.dims_per_head,
            bias=self.use_qkv_bias,
        )
        self.k_for_heads = nn.Linear(
            in_features=self.in_out_dims,
            out_features=self.num_heads * self.dims_per_head,
            bias=self.use_qkv_bias,
        )
        self.v_for_heads = nn.Linear(
            in_features=self.in_out_dims,
            out_features=self.num_heads * self.dims_per_head,
            bias=self.use_qkv_bias,
        )
        # 最后调整输出维度的MLP
        self.linear_projection = nn.Linear(
            in_features=self.dims_per_head * self.num_heads,
            out_features=self.in_out_dims,
        )

        # 位置编码相关
        self.relative_position_bias_indices = (
            self.get_initial_relative_position_indices()
        )
        # self.register_buffer(
        #     name="relative_position_bias_indices", tensor=relative_position_bias_indices
        # )

        # 位置编码的表。需要从中查表取值。
        relative_position_bias_table_shape: tuple = (
            2 * self.window_size[0] - 1,
            2 * self.window_size[1] - 1,
        )
        table = torch.randn(
            size=(
                relative_position_bias_table_shape[0],
                relative_position_bias_table_shape[1],
            )
        )
        self.relative_position_bias_table = nn.Parameter(data=table, requires_grad=True)
        # 测试用：self.relative_position_bias_table = get_distance_code()

        # cyclic_shift的mask相关。保存mask_for_cyclic_shift。初始化为None，当self.feature_shape_hw被初始化之后才能计算。
        self.mask_for_cyclic_shift: Tensor = torch.tensor([])
        # self.register_buffer(name="mask_for_cyclic_shift", tensor=None)

    def initialize_feature_shape_hw(self, q):
        """q shape is (batch_size, channels, height, width)"""
        _, _, h, w = q.shape
        if self.training:  # 训练时只需记录形状一次，每个输入形状相同
            if (
                not self.feature_shape_hw
            ):  # 空tuple判定为False，用not转为True。所以空tuple时进入if
                self.feature_shape_hw = (h, w)
        else:
            self.feature_shape_hw = (h, w)

    def get_initial_relative_position_indices(self):
        """Return an indices tensor, with shape (x, y, 2).
        Where x and y are both "win_size_h * win_size_w".
        The last dim of length 2 tells the coordinates to access in bias table.
        The coordinates mean relative position vectors.
        Bias table saves bias values.
        This will help to add bias for attention scores.
        Note that these indices only depend on window size and are initialized along with the __init__().
        In the training process, it stays unchanged. That's why it is registered as a buffer.

        Returns:
            Tensor
        """
        meshgrid = torch.stack(
            tensors=torch.meshgrid(
                [torch.arange(self.window_size[0]), torch.arange(self.window_size[1])],
                indexing="ij",
            ),
            dim=0,
        )  # (2, h, w)
        meshgrid = meshgrid.flatten(start_dim=1)
        # (2, 1, h*w) - (2, h*w, 1)
        relative_position_indices = meshgrid[:, None, :] - meshgrid[:, :, None]
        relative_position_indices[0, :, :] += self.window_size[0] - 1
        relative_position_indices[1, :, :] += self.window_size[1] - 1
        return relative_position_indices

    def get_new_relative_position_bias(self):
        """Return a tensor with the same shape as attention scores.
        Shape is (num_points_per_window, num_points_per_window).
        This function should be called every time when forward() is called, because relative position bias
        is a learnable parameter, and it updates every iteration while training.
        You must fetch the latest bias by calling this.

        Returns:
            Tensor
        """
        edge_length = self.window_size[0] * self.window_size[1]
        # relative_position_bias_indices shape is (2, edge_length, edge_length)
        indices = self.relative_position_bias_indices.reshape(2, -1)
        x_indices, y_indices = indices[0], indices[1]
        # bias shape is 1 dim, len is (edge_length * edge_length)
        bias = self.relative_position_bias_table[x_indices, y_indices]
        # reshape to 2 dim, (edge_length, edge_length)
        return bias.reshape(edge_length, edge_length)

    def check_qkv(self, q, k, v):
        R"""
        Check if given qkv is aligned with the option self.use_cross_attention.
        If the mode is cross, k and v should be the same, and q should be different.
        If the mode is self, q, k, and v should all be the same.
        """
        pass

    def rearrange_1(self, tensor: Tensor):
        R"""Change a tensor shape by:
        b c (n_win_h win_size_h) (n_win_w win_size_w) -> (b n_win_h n_win_w) (win_size_h win_size_w) c
        Could be understood as window partition operation.

        Args:
            tensor (Tensor): original q, k, or v passed into forward()

        Returns:
            Tensor
        """
        return einops.rearrange(
            tensor=tensor,
            pattern="b c (n_win_h win_size_h) (n_win_w win_size_w) -> (b n_win_h n_win_w) (win_size_h win_size_w) c",
            n_win_h=self.feature_shape_hw[0] // self.window_size[0],
            n_win_w=self.feature_shape_hw[1] // self.window_size[1],
            win_size_h=self.window_size[0],
            win_size_w=self.window_size[1],
        )

    def rearrange_2(self, tensor: Tensor):
        R"""Change a tensor shape by:
        B t (num_heads dims_per_head) -> B num_heads t dims_per_head
        Split heads from channels.
        Where B is "batch_size *
                    n_wins_per_image_in_height_direction *
                    n_wins_per_image_in_width_direction".
        And t is num of points per window, which is "win_size_h * win_size_w"

        Args:
            tensor (Tensor)

        Returns:
            Tensor
        """
        return einops.rearrange(
            tensor=tensor,
            pattern="B t (num_heads dims_per_head) -> B num_heads t dims_per_head",
            num_heads=self.num_heads,
            dims_per_head=self.dims_per_head,
        )

    def create_heads_for_qkv(self, q: Tensor, k: Tensor, v: Tensor):
        R"""Create multiple heads. Return shape (B, n_heads, t, dims_per_head).
        Where B is "batch_size * n_wins_per_image_in_height_direction * n_wins_per_image_w"
        And t is num of points per window, which is "win_size_h * win_size_w"

        Args:
            q (Tensor): shape is (batch_size, channels, height, width)
            k (Tensor): same shape as q
            v (Tensor): same shape as q

        Returns:
            Tensor: shape is (B, n_heads, t, dims_per_head)
        """
        # window partition
        q, k, v = [self.rearrange_1(elem) for elem in [q, k, v]]
        # linear layer
        q, k, v = self.q_for_heads(q), self.k_for_heads(k), self.v_for_heads(v)
        # pick out heads from the channel dimension
        q, k, v = [self.rearrange_2(elem) for elem in [q, k, v]]
        return q, k, v

    def initialize_mask_for_cyclic_shift(self):
        R"""Initialize attribute self.mask_for_cyclic_shift.
        Shape will be 3D, (n_wins_per_image, n_points_per_win, n_points_per_win), filled with boolean.
        Next, -inf should be plugged into Ture position.
        """
        win_size_h, win_size_w = self.window_size
        shift_size_h, shift_size_w = win_size_h // 2, win_size_w // 2

        slices_h = [
            slice(0, -win_size_h, 1),
            slice(-win_size_h, -shift_size_h, 1),
            slice(-shift_size_h, None, 1),
        ]
        slices_w = [
            slice(0, -win_size_w, 1),
            slice(-win_size_w, -shift_size_w, 1),
            slice(-shift_size_w, None, 1),
        ]

        assert (
            self.feature_shape_hw is not None
        ), """
        Attribute "self.feature_shape_hw" is None.
        Check why function "self.initialize_feature_shape_hw()" is not called before this.
        """
        image_region_hint = torch.zeros(size=self.feature_shape_hw, dtype=torch.int)
        cnt = 0
        for slice_h in slices_h:
            for slice_w in slices_w:
                image_region_hint[slice_h, slice_w] = cnt
                cnt += 1

        # 借助self.rearrange_1()方法，做window partition。但是需要输入形状改变为方法要求的（batch_size, c, h, w）
        # 缺少两个维度，用None添加。也可以用unsqueeze()。
        image_region_hint = image_region_hint[None, None, :, :]
        # 返回的形状为"(b n_win_h n_win_w) (win_size_h win_size_w) c"
        # 也即3D, (n_win_h * n_win_w, win_size_h * win_size_w, 1)
        image_region_hint = self.rearrange_1(image_region_hint)
        # squeeze掉最后一个维度，shape变为(n_wins_per_image, n_points_per_image)
        image_region_hint = image_region_hint.squeeze(dim=-1)

        # 对points分别排成一列和一行，再分别广播至2D，得到attention score相同的矩阵形状。
        # attention score矩阵含义是，一行代表一个点，它与各个列的点的注意力分数。
        # 拥有不同region编号cnt的位置，置为False。相同处置为True。
        one_column = image_region_hint[:, :, None]
        one_row = image_region_hint[:, None, :]

        expected_shape = (
            image_region_hint.shape[0],
            image_region_hint.shape[1],
            image_region_hint.shape[1],
        )
        one_column, one_row = one_column.expand(expected_shape), one_row.expand(
            expected_shape
        )
        self.mask_for_cyclic_shift = torch.ne(one_column, one_row)

    def operate_mask_scores_for_cyclic_shift(self, scores: Tensor):
        """Use self.mask_for_cyclic_shift(mask) to mask scores.
        Mask contains full bool values, with shape (n_wins_per_image, t, t).
        "scores" has shape (batch_size * n_wins_per_image, n_heads, t, t).
        Here, t = n_points_per_win, and B = batch_size * n_wins_per_image.

        First, mask should be broadcast to the same shape as scores.
        Next, use mask as indices to set those True positions of scores to -inf.

        Args:
            scores (_type_): _description_
        """
        # 如果self.mask_for_cyclic_shift为None，说明还没有初始化，是首次iteration。需要进行初始化。
        if self.mask_for_cyclic_shift.numel() == 0:
            self.initialize_mask_for_cyclic_shift()
        mask = self.mask_for_cyclic_shift

        n_wins_per_image = mask.shape[0]
        # scores每个图的窗口个数的维度单独提取出来，避免mask使用repeat占用更多内存。因为长度非1的维度不能广播。
        scores = einops.rearrange(
            tensor=scores,
            pattern="(b nw) nh t1 t2 -> b nw nh t1 t2",  # b为batch_size, nh为n_heads。t1=t2, einops要求不能重名t
            nw=n_wins_per_image,
        )

        # scores改变形状后，mask可以广播。先添加维度，mask缺少b和n_heads维度。
        # 另一种方法是计算全程都保持scores的n_wins_per_image是一个单独的维度，这样就要修改self.arrange_1()等方法
        # (nw, t, t) -> (1, nw, 1, t, t)
        mask = mask.unsqueeze(1).unsqueeze(0)
        # mask shape (1, nw, 1, t, t) -> (b, nw, nh, t, t)
        mask = mask.expand_as(scores)

        # scores根据mask改值。然后reshape回到输入时的形状。
        # scores[mask] = torch.finfo(torch.float32).min
        scores[mask] = -1e10
        scores = einops.rearrange(
            tensor=scores,
            pattern="b nw nh t1 t2 -> (b nw) nh t1 t2",  # b为batch_size, nh为n_heads;
        )
        return scores

    def calculate_attention(self, q, k, v):
        R"""
        1. Calculate the score.
        2. Calculate the attention weights.
        3. Calculate the attention output.

        Args:
            q (Tensor): shape is (B, n_heads, t, dims_per_head)
            k (Tensor): same shape as q
            v (Tensor): same shape as q

        Returns:
            Tensor: shape is (B, n_heads, t, dims_per_head)
        """

        """1. Calculate attention scores."""
        k = k.permute(0, 1, 3, 2)
        # scores shape is (B, n_heads, t, t)
        scores = torch.matmul(input=q, other=k) * self.qk_scale

        """2. Add bias"""
        # got bias has shape (t, t). Need to expand to the same as scores (B, n_heads, t, t)
        bias = self.get_new_relative_position_bias()
        bias = bias[None, None, :, :].expand_as(scores)
        scores += bias

        """3. Operate mask if shift is enabled."""
        if self.use_cyclic_shift:
            scores = self.operate_mask_scores_for_cyclic_shift(scores)

        """4. Calculate the attention weights."""
        # shape (B, n_heads, t, t)
        weights = f.softmax(input=scores, dim=-1)

        """5. Get out_values. Dropout."""
        # v shape (B, n_heads, t, dims_per_head)
        out_values = torch.matmul(input=weights, other=v)
        return self.attention_drop_layer(out_values)

    @staticmethod
    def rearrange_2_reverse(tensor: Tensor):
        """Merge n_heads dimension into channels dimension.
        Do the opposite thing to rearrange_2().
        (B, n_heads, t, dims_per_head) -> (B, n_heads * dims_per_head, t)

        Args:
            tensor (Tensor): 4D, (B, n_heads, t, dims_per_head)

        Returns:
            tensor (Tensor): 3D, (B, t, n_heads * dims_per_head)
        """
        return einops.rearrange(
            tensor=tensor,
            pattern="B n_heads t dims_per_head -> B t (n_heads dims_per_head)",
        )

    def rearrange_1_reverse(self, tensor: Tensor):
        """Merge windows back to image shape.

        Args:
            tensor (Tensor): shape 3D, (B, t, out_dims), where B = batch_size * n_wins_h * n_wins_w,
                                                               t = win_size_h * win_size_w

        Returns:
            Tensor: 4D, (batch_size, channels, image_shape_h, image_shape_w),
                where channels = out_dims, image_shape_h = n_wins_h * win_size_h,
                image_shape_w = n_wins_w * win_size_w.
        """
        win_size_h, win_size_w = self.window_size
        n_wins_h, n_wins_w = (
            self.feature_shape_hw[0] // win_size_h,
            self.feature_shape_hw[1] // win_size_w,
        )
        return einops.rearrange(
            tensor=tensor,
            pattern="(batch_size n_wins_h n_wins_w) (win_size_h win_size_w) out_dims -> "
            "batch_size out_dims (n_wins_h win_size_h) (n_wins_w win_size_w)",
            n_wins_h=n_wins_h,
            n_wins_w=n_wins_w,
            win_size_h=win_size_h,
            win_size_w=win_size_w,
        )

    def calculate_linear_projection_and_reshape_back(self, tensor: Tensor):
        """Main purpose is to change num of channels to you want.
        It is set by "self.out_dims", often taken the same as "self.in_dims".

        Args:
            tensor (Tensor): shape (B, n_heads, t, dims_per_head),
                where t = n_points_per_win, and B = batch_size * n_wins_per_image.

        Returns:
            Tensor: 4D, (batch_size, channels, image_shape_h, image_shape_w)
        """
        tensor = self.rearrange_2_reverse(tensor)
        # projection changes dim (B, t, n_heads * dims_per_head) -> (B, t, out_dims)
        tensor = self.linear_projection(tensor)
        tensor = self.linear_drop_layer(tensor)
        # change back to original shape (B, t, out_dims) -> (batch_size, channels, image_shape_h, image_shape_w)
        tensor = self.rearrange_1_reverse(tensor)
        return tensor

    def check_if_need_and_do_or_undo_cyclic_shift(self, tensor_list, mode: str) -> list:
        """
        Args:
            tensor_list: a list of tensors, each one is shape 4D, (batch_size, channels, height, width)
            mode: "do" or "undo"

        Returns:
            tensor list after cyclic shift.
        """
        # 如果不需要cyclic shift，直接返回
        if not self.use_cyclic_shift:
            return tensor_list
        # 需要cyclic shift，计算
        win_size_h, win_size_w = self.window_size
        shift_size_h, shift_size_w = win_size_h // 2, win_size_w // 2

        if mode == "do":
            shift_size_h, shift_size_w = -shift_size_h, -shift_size_w
        elif mode == "undo":
            pass
        else:
            raise ValueError(f"{mode} is not supported.")

        tensor_list = [
            torch.roll(input=elem, shifts=(shift_size_h, shift_size_w), dims=(2, 3))
            for elem in tensor_list
        ]
        return tensor_list

    def forward(self, q, k, v):
        """
        Args:
            q: shape 4D, (batch_size, channels, height, width)
            k: same shape as q
            v: same shape as q

        Returns: Tensor 4D, (batch_size, out_dims, height, width)

        """
        """准备"""
        self.check_qkv(q, k, v)
        self.initialize_feature_shape_hw(q)
        """cyclic shift，如果需要。"""
        q, k, v = self.check_if_need_and_do_or_undo_cyclic_shift(
            tensor_list=[q, k, v], mode="do"
        )
        """计算"""
        q, k, v = self.create_heads_for_qkv(q, k, v)
        output = self.calculate_attention(q, k, v)
        output = self.calculate_linear_projection_and_reshape_back(output)
        """undo cyclic shift，如果需要。"""
        # 只有一个tensor时也返回list，所以要取list的第0项。
        output = self.check_if_need_and_do_or_undo_cyclic_shift(
            tensor_list=[output], mode="undo"
        )[0]
        return output


def test_attention():
    batch_size = 5
    channels = 3
    h = 49
    w = 49

    test_tensor = torch.zeros(size=(batch_size, channels, h, w))

    window_attention = WindowAttention(
        in_out_dims=3,
        num_heads=4,
        dims_per_head=3,
        window_size=(7, 7),
        use_cyclic_shift=True,
        use_cross_attention=False,
        use_qkv_bias=True,
        attention_drop_ratio=0,
        linear_after_att_drop_ratio=0,
    )

    # plt.imshow(window_attention.get_new_relative_position_bias(), cmap="gray")
    # plt.show()

    for _ in tqdm(range(10000)):
        window_attention(q=test_tensor, k=test_tensor, v=test_tensor)
    print(test_tensor.shape)


# def test_iter():
#     for _ in range():
#         pass

if __name__ == "__main__":
    test_attention()
