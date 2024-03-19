import einops
import torch
from torch import Tensor
from tqdm import tqdm


def put_channel_dim_to_the_last_position(tensor: Tensor):
    """
    Args:
        tensor: shape 4D, (batch_size, channels, h, w)

    Returns:
        shape 4D, (batch_size, h, w, channels)
    """
    return tensor.permute(dims=(0, 2, 3, 1))


def put_channel_dim_to_the_second_position(tensor: Tensor):
    """
    Args:
        tensor: shape 4D, (batch_size, h, w, channels)

    Returns:
        shape 4D, (batch_size, channels, h, w)
    """
    return tensor.permute(dims=(0, 3, 1, 2))


