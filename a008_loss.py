import torch
from kornia.losses import ssim_loss
from torch import Tensor, nn
from torch.nn import functional
from tqdm import tqdm


class MyLoss(nn.Module):
    """
    This class implements a custom loss function for fusion images.
    The loss function combines SSIM loss, texture loss, and intensity loss.
    """

    def __init__(self):
        super().__init__()
        # ssim
        self.ssim_loss_window_size = 11
        self.max_val = 1.0
        self.fus_ir_similarity_weight = 0.4  # important
        self.fus_vis_similarity_weight = 1 - self.fus_ir_similarity_weight
        # texture
        self.init_texture_utils()
        # intensity loss has nothing to register as attributes

        # merging weight to produce total weight, important
        self.ssim_loss_ratio = 0.3
        self.texture_loss_ratio = 0.3
        self.intensity_loss_ratio = 0.4

    def init_texture_utils(self):
        """
        This function initializes the Sobel kernel used for texture loss.
        The Sobel kernel is a 3x3 kernel that is used to calculate the gradient of the image.
        """
        sobel_kernel_x = torch.tensor(
            data=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float
        )[None, None, :, :]
        sobel_kernel_y = torch.tensor(
            data=[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float
        )[None, None, :, :]
        self.register_buffer(name="sobel_kernel_x", tensor=sobel_kernel_x)
        self.register_buffer(name="sobel_kernel_y", tensor=sobel_kernel_y)

    def calcu_ssim_loss(
        self,
        fusion_images: Tensor,
        ir_images: Tensor,
        vis_images: Tensor,
    ) -> Tensor:
        """
        This function calculates the SSIM loss between the fusion images, IR images, and visible images.
        The SSIM loss is calculated using the ssim_loss function from the kornia library.
        The SSIM loss is weighted based on the importance of IR and visible images.

        Args:
            fusion_images: The fusion images, a tensor of shape (batch_size, channels, height, width).
            ir_images: The IR images, a tensor of the same shape as fusion_images.
            vis_images: The visible images, a tensor of the same shape as fusion_images.

        Returns:
            The SSIM loss, a tensor of shape (1,).
        """
        fus_vis_loss = 2 * ssim_loss(  # 默认ssim_loss公式是(1-ssim)/2
            img1=fusion_images,
            img2=vis_images,
            window_size=self.ssim_loss_window_size,
            max_val=self.max_val,
            reduction="mean",
            padding="same",
        )
        fus_ir_loss = 2 * ssim_loss(
            img1=fusion_images,
            img2=ir_images,
            window_size=self.ssim_loss_window_size,
            max_val=self.max_val,
            reduction="mean",
            padding="same",
        )
        return (
            fus_ir_loss * self.fus_ir_similarity_weight
            + fus_vis_loss * self.fus_vis_similarity_weight
        )

    def sobel_grad_for_one_image(self, image: Tensor) -> Tensor:
        """
        This function calculates the Sobel gradient of an image.
        The Sobel gradient is calculated using a 3x3 Sobel kernel.
        The gradient is calculated for each channel independently, and then combined into a single tensor.

        Args:
            image: The input image, a tensor of shape (batch_size, channels, height, width).

        Returns:
            The Sobel gradient, a tensor of shape (batch_size, 1, height, width).
        """
        channels = image.shape[1]
        # 卷积核形状为(全部核数量(输出通道总数), in_channels//groups, kernel_size_h, kernel_size_w)
        # 分组的意思是，feature的所有channels深度分为几片。每一片只分配几个核，而不是所有的核。
        # in_channels//groups是指，对feature的深度分组后，每个组的深度，与每个核的深度匹配。
        # 这里sobel_kernel_x (1, 1, kernel_h, kernel_w) -> (1, channels, kernel_h, kernel_w)
        sobel_kernel_x = self.sobel_kernel_x.expand(1, channels, -1, -1)
        sobel_kernel_y = self.sobel_kernel_y.expand(1, channels, -1, -1)
        # 卷积结果将等于 (batch, out_channels, image_h, image_w)
        grad_x = functional.conv2d(
            input=image, weight=sobel_kernel_x, padding=1, groups=1
        )
        grad_y = functional.conv2d(
            input=image, weight=sobel_kernel_y, padding=1, groups=1
        )
        return torch.abs(grad_x) + torch.abs(grad_y)

    def calcu_texture_loss(
        self,
        fusion_images: Tensor,
        ir_images: Tensor,
        vis_images: Tensor,
    ) -> Tensor:
        """
        This function calculates the texture loss between the fusion images, IR images, and visible images.
        The texture loss is calculated using the Sobel gradient of the images.
        The texture loss is normalized by the number of pixels in the image.

        Args:
            fusion_images: The fusion images, a tensor of shape (batch_size, channels, height, width).
            ir_images: The IR images, a tensor of the same shape as fusion_images.
            vis_images: The visible images, a tensor of the same shape as fusion_images.

        Returns:
            The texture loss, a tensor of shape (1,).
        """
        # grad shape is 4D, (b, 1, h, w)
        grad_fus, grad_ir, grad_vis = (
            self.sobel_grad_for_one_image(fusion_images),
            self.sobel_grad_for_one_image(ir_images),
            self.sobel_grad_for_one_image(vis_images),
        )
        b, _, h, w = grad_fus.shape

        result = grad_fus - torch.max(grad_ir, grad_vis)
        result = torch.norm(input=result, p=1, keepdim=False) / result.numel()
        return result

    @staticmethod
    def calcu_intensity_loss(
        fusion_images: Tensor,
        ir_images: Tensor,
        vis_images: Tensor,
    ) -> Tensor:
        """
        This function calculates the intensity loss between the fusion images, IR images, and visible images.
        The intensity loss is calculated as the L1 norm between the
            fusion images and the maximum of the IR and visible images.

        Args:
            fusion_images: The fusion images, a tensor of shape (batch_size, channels, height, width).
            ir_images: The IR images, a tensor of the same shape as fusion_images.
            vis_images: The visible images, a tensor of the same shape as fusion_images.

        Returns:
            The intensity loss, a tensor of shape (1,).
        """
        max_value = torch.max(ir_images, vis_images)
        return (
            torch.norm(input=fusion_images - max_value, p=1, keepdim=False)
            / fusion_images.numel()
        )

    def calcu_total_loss(
        self,
        fusion_images: Tensor,
        ir_images: Tensor,
        vis_images: Tensor,
    ) -> Tensor:
        """
        Calculates the total loss for the fusion images, given the IR and visible images.

        Args:
            fusion_images: The fusion images, a tensor of shape (batch_size, channels, height, width).
            ir_images: The IR images, a tensor of the same shape as fusion_images.
            vis_images: The visible images, a tensor of the same shape as fusion_images.

        Returns:
            The total loss, a tensor of shape (1,).
        """
        # calculate SSIM loss
        the_ssim_loss = self.calcu_ssim_loss(fusion_images, ir_images, vis_images)

        # calculate texture loss
        texture_loss = self.calcu_texture_loss(fusion_images, ir_images, vis_images)

        # calculate intensity loss
        intensity_loss = self.calcu_intensity_loss(fusion_images, ir_images, vis_images)

        # calculate total loss
        total_loss = (
            the_ssim_loss * self.ssim_loss_ratio
            + texture_loss * self.texture_loss_ratio
            + intensity_loss * self.intensity_loss_ratio
        )

        return total_loss


def test_my_ssim_loss():
    my_loss = MyLoss()
    img1 = torch.randn(size=(4, 6, 224, 224))
    img2 = torch.randn(size=(4, 6, 224, 224))
    img3 = torch.randn(size=(4, 6, 224, 224))
    print(my_loss.calcu_ssim_loss(img1, img2, img3))


def test_texture_loss():
    my_loss = MyLoss()
    img1 = torch.randn(size=(4, 6, 224, 224))
    img2 = torch.randn(size=(4, 6, 224, 224))
    img3 = torch.randn(size=(4, 6, 224, 224))
    print(my_loss.calcu_texture_loss(img1, img2, img3))


def test_total_loss():
    my_loss = MyLoss().cuda()
    img1 = torch.randn(size=(4, 6, 224, 224)).cuda()
    img2 = torch.randn(size=(4, 6, 224, 224)).cuda()
    img3 = torch.randn(size=(4, 6, 224, 224)).cuda()

    for _ in tqdm(range(100000)):
        my_loss.calcu_total_loss(img1, img2, img3)


if __name__ == "__main__":
    test_total_loss()
