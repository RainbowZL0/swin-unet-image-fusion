"""
类似于把几张图片在高宽上拼成一个大图。
"""
import torch
import torchvision
from glob import glob
import einops

IMAGE_FOLDER = R".\_test_images"


def test_cuda():
    print(torch.cuda.get_device_name(0))
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("cuDNN Enabled:", torch.backends.cudnn.enabled)
    print("cuDNN Version:", torch.backends.cudnn.version())
    print(torch.cuda_version)


def start():
    path_list = glob(IMAGE_FOLDER + r"\*.jpg")
    # torch读入图片后格式为(c, h, w)
    image_tensor_list = [torchvision.io.read_image(path)
                         for path in path_list]

    # four_image_tensor的shape为(4, c, h, w)
    four_image_tensor = torch.stack(image_tensor_list)

    torchvision.io.write_jpeg(method_1(four_image_tensor),
                              r".\test_1.jpg")
    torchvision.io.write_jpeg(method_2(four_image_tensor),
                              r".\test_2.jpg")


def method_1(four_image_tensor):
    """
    仅用reshape和permute完成
    """
    _, c, h, w = four_image_tensor.shape
    four_image_tensor = four_image_tensor.reshape(2, 2, c, h, w)
    four_image_tensor = four_image_tensor.permute(2, 0, 3, 1, 4)
    four_image_tensor = four_image_tensor.reshape(c, 2 * h, 2 * w)
    return four_image_tensor


def method_2(four_image_tensor):
    """
    借助einops
    """
    _, c, h, w = four_image_tensor.shape
    four_image_tensor = einops.rearrange(tensor=four_image_tensor,
                                         pattern="(n_h n_w) c h w -> c (n_h h) (n_w w)",
                                         n_h=2,
                                         n_w=2)
    return four_image_tensor


if __name__ == "__main__":
    start()
