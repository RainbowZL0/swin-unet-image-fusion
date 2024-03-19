from collections import deque

import torch
import numpy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

DEVICE = torch.device("cuda")


def start():
    a = torch.zeros(size=(7, 7))
    distance_weight = get_distance_code()

    bias = torch.zeros(size=(49, 49))
    encoding_position = torch.zeros(size=(2, 49, 49), dtype=torch.int)

    indices_tensor = torch.tensor(
        data=numpy.indices(dimensions=(49, 49), dtype=int), dtype=torch.int
    )

    encoding_position[0, :, :] = indices_tensor[1] // 7 - indices_tensor[0] // 7 + 6
    encoding_position[1, :, :] = indices_tensor[1] % 7 - indices_tensor[0] % 7 + 6

    # # 这样的写法不允许，因为右端不是逐个完成的，而是全部的。
    # encoding_position = encoding_position.permute(1, 2, 0)  # (49, 49, 2)
    # bias[all_indices_list] = encoding[
    #     encoding_position[all_indices_list][0],
    #     encoding_position[all_indices_list][1]
    # ]

    # 方法1 reshape
    now = time.time()
    for epoch in tqdm(range(1000000)):
        bias_copy_0 = bias.clone()
        encoding_position_copy_0 = encoding_position.clone()

        row_indices = encoding_position_copy_0[0].reshape(-1)  # 1 dim, (49*49)
        col_indices = encoding_position_copy_0[1].reshape(-1)

        bias_copy_0 = bias_copy_0.reshape(-1)

        bias_copy_0[:] = distance_weight[row_indices, col_indices]

        bias_copy_0 = bias_copy_0.reshape(49, 49)
    print(f"reshape 1000次，耗时 {time.time() - now}")
    # plt.imshow(bias_copy_0.numpy(), cmap="gray")
    # plt.show()
    # pass

    # # 方法2 循环
    # now = time.time()
    # for epoch in range(1000):
    #     bias_copy_1 = bias.clone()
    #     encoding_position_copy_1 = encoding_position.clone().permute(1, 2, 0)
    #     for i in range(49):
    #         for j in range(49):
    #             encoding_row_col = encoding_position_copy_1[i, j]
    #             bias_copy_1[i, j] = distance_weight[*encoding_row_col]
    # print(f"循环 1000次，耗时 {time.time() - now}")
    # # plt.imshow(bias_copy_1, cmap="gray")
    # # plt.show()

    # # 方法3 map()
    # now = time.time()
    # bias_copy_2 = bias.clone()
    # for epoch in range(1000):
    #     encoding_position_copy_2 = encoding_position.clone().permute(1, 2, 0)
    #     indices = torch.tensor(
    #         numpy.indices(dimensions=(49, 49), dtype=int), dtype=torch.int
    #     )
    #     indices = indices.permute(1, 2, 0)
    #     indices = indices.reshape(-1, 2).tolist()

    #     def operate(elem):
    #         """
    #         @param elem: [row, col]
    #         @return:
    #         """
    #         row, col = tuple(elem)
    #         weight_row, weight_col = tuple(encoding_position_copy_2[row, col].tolist())
    #         bias_copy_2[row, col] = distance_weight[weight_row, weight_col]
    #         return None

    #     deque(map(operate, indices), maxlen=0)
    # print(f"map 1000次，耗时 {time.time() - now}")

    # plt.imshow(bias_copy_2, cmap="gray")
    # plt.show()
    # pass


def get_distance_code():
    indices = torch.tensor(
        numpy.indices(dimensions=(13, 13), dtype=int), dtype=torch.int
    )
    indices -= 6
    row_indices, col_indices = indices[0, :], indices[1, :]
    distance_code = torch.sqrt(torch.pow(row_indices, 2) + torch.pow(col_indices, 2))
    distance_weight = torch.pow(distance_code, -0.8).nan_to_num(posinf=1)

    plt.imshow(distance_weight, cmap="gray")
    plt.show()

    # print(distance_weight)

    return distance_weight


if __name__ == "__main__":
    start()
