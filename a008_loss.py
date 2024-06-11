import numpy as np
import torch
from kornia.losses import ssim_loss, MS_SSIMLoss, PSNRLoss
from kornia.filters import Canny, Sobel
from torch import Tensor, nn
from torch.nn import functional
from tqdm import tqdm

import A000_CONFIG as MyConfig
from a010_StateRecorder import StateRecorder


class MyLoss(nn.Module):
    """
    This class implements a custom loss function for fusion images.
    The loss function combines SSIM loss, texture loss, and intensity loss.
    """

    def __init__(self):
        super().__init__()
        # ssim
        self.use_multi_scale_ssim = MyConfig.CHOOSE_MS_SSIM
        if self.use_multi_scale_ssim:
            self.ms_ssim_func = MS_SSIMLoss()
        else:
            self.ssim_func = ssim_loss
            self.ssim_loss_window_size = 11
        self.max_val = 1.0
        self.fus_ir_ssim_weight = MyConfig.FUS_IR_SSIM_WEIGHT  # important
        self.fus_vis_ssim_weight = 1 - self.fus_ir_ssim_weight

        # texture
        self.choose_canny = MyConfig.CHOOSE_CANNY_ELSE_SOBEL
        if self.choose_canny:
            self.texture_func = Canny()
        else:
            self.texture_func = Sobel()
        # self.init_texture_utils()

        # intensity loss has nothing to register as attributes

        # 峰值信噪比 psnr
        self.use_psnr = MyConfig.USE_PSNR
        if self.use_psnr:
            self.psnr_func = PSNRLoss(max_val=1.0)
            self.fus_ir_psnr_weight = MyConfig.FUS_IR_PSNR_WEIGHT
            self.fus_vis_psnr_weight = 1 - self.fus_ir_psnr_weight

        # 放缩loss的倍数，使得他们有接近的数量级。其实和下面的merging weight的作用类似，只是有不同的含义
        self.ssim_scale = MyConfig.SSIM_SCALE
        self.texture_scale = MyConfig.TEXTURE_SCALE
        self.intensity_scale = MyConfig.INTENSITY_SCALE
        self.psnr_scale = MyConfig.PSNR_SCALE
        # 假设三个loss已经有相同的数量级，加权平均成为总loss。merging weight to produce total weight, important
        self.ssim_loss_ratio = MyConfig.SSIM_LOSS_RATIO
        self.texture_loss_ratio = MyConfig.TEXTURE_LOSS_RATIO
        self.intensity_loss_ratio = MyConfig.INTENSITY_LOSS_RATIO
        self.psnr_loss_ratio = MyConfig.PSNR_LOSS_RATIO

        # 记录loss。两个记录器都将会封装字典作为元素
        self.loss_recorder_in_detail = StateRecorder()  # 每次计算的loss都会存入，在适当的时机计算一次平均值，然后清空
        self.mean_loss_recorder = StateRecorder()  # 计算的平均值将存入

    def calcu_psnr_loss(
            self,
            fusion_images: Tensor,
            ir_images: Tensor,
            vis_images: Tensor,
    ):
        fus_ir_loss = self.psnr_func(fusion_images, ir_images)
        fus_vis_loss = self.psnr_func(fusion_images, vis_images)
        return (self.fus_ir_psnr_weight * fus_ir_loss
                + self.fus_vis_psnr_weight * fus_vis_loss)

    # def init_texture_utils(self):
    #     """
    #     This function initializes the Sobel kernel used for texture loss.
    #     The Sobel kernel is a 3x3 kernel that is used to calculate the gradient of the image.
    #     """
    #     sobel_kernel_x = torch.tensor(
    #         data=[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float
    #     )[None, None, :, :]
    #     sobel_kernel_y = torch.tensor(
    #         data=[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float
    #     )[None, None, :, :]
    #     self.register_buffer(name="sobel_kernel_x", tensor=sobel_kernel_x)
    #     self.register_buffer(name="sobel_kernel_y", tensor=sobel_kernel_y)

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
        if self.use_multi_scale_ssim:
            fus_ir_loss = self.ms_ssim_func(fusion_images, ir_images)
            fus_vis_loss = self.ms_ssim_func(fusion_images, vis_images)
        else:
            fus_ir_loss = 2 * self.ssim_func(
                img1=fusion_images,
                img2=ir_images,
                window_size=self.ssim_loss_window_size,
                max_val=self.max_val,
                reduction="mean",
                padding="same",
            )
            fus_vis_loss = 2 * self.ssim_func(  # 默认ssim_loss公式是(1-ssim)/2
                img1=fusion_images,
                img2=vis_images,
                window_size=self.ssim_loss_window_size,
                max_val=self.max_val,
                reduction="mean",
                padding="same",
            )
        return (
                fus_ir_loss * self.fus_ir_ssim_weight
                + fus_vis_loss * self.fus_vis_ssim_weight
        )

    # def sobel_grad_for_one_image(self, image: Tensor) -> Tensor:
    #     """
    #     This function calculates the Sobel gradient of an image.
    #     The Sobel gradient is calculated using a 3x3 Sobel kernel.
    #     The gradient is calculated for each channel independently, and then combined into a single tensor.
    #
    #     Args:
    #         image: The input image, a tensor of shape (batch_size, channels, height, width).
    #
    #     Returns:
    #         The Sobel gradient, a tensor of shape (batch_size, 1, height, width).
    #     """
    #     channels = image.shape[1]
    #     # 卷积核形状为(全部核数量(输出通道总数), in_channels//groups, kernel_size_h, kernel_size_w)
    #     # 分组的意思是，feature的所有channels深度分为几片。每一片只分配几个核，而不是所有的核。
    #     # in_channels//groups是指，对feature的深度分组后，每个组的深度，与每个核的深度匹配。
    #     # 这里sobel_kernel_x (1, 1, kernel_h, kernel_w) -> (1, channels, kernel_h, kernel_w)
    #     sobel_kernel_x = self.sobel_kernel_x.expand(1, channels, -1, -1)
    #     sobel_kernel_y = self.sobel_kernel_y.expand(1, channels, -1, -1)
    #     # 卷积结果将等于 (batch, out_channels, image_h, image_w)
    #     grad_x = functional.conv2d(
    #         input=image, weight=sobel_kernel_x, padding=1, groups=1
    #     )
    #     grad_y = functional.conv2d(
    #         input=image, weight=sobel_kernel_y, padding=1, groups=1
    #     )
    #     return torch.abs(grad_x) + torch.abs(grad_y)

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
        使融合的图片有ir和vis二者中更多的边缘细节
        Args:
            fusion_images: The fusion images, a tensor of shape (batch_size, channels, height, width).
            ir_images: The IR images, a tensor of the same shape as fusion_images.
            vis_images: The visible images, a tensor of the same shape as fusion_images.

        Returns:
            The texture loss, a tensor of shape (1,).
        """
        # shape is 4D, (b, 1, h, w)
        # edge_fus, edge_ir, edge_vis = (
        #     self.sobel_grad_for_one_image(fusion_images),
        #     self.sobel_grad_for_one_image(ir_images),
        #     self.sobel_grad_for_one_image(vis_images),
        # )
        edge_result_list = []
        for feature in [fusion_images, ir_images, vis_images]:
            if self.choose_canny:
                _, result = self.texture_func(feature)
            else:
                result = self.texture_func(feature)
            edge_result_list.append(result)

        edge_fus, edge_ir, edge_vis = edge_result_list
        b, _, h, w = edge_fus.shape

        result = edge_fus - torch.max(edge_ir, edge_vis)
        result = torch.mean(torch.abs(result))
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
        使得融合图片容易获得ir和vis中更亮的部分
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
    ):
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
        the_ssim_loss = (self.calcu_ssim_loss(fusion_images, ir_images, vis_images)
                         * self.ssim_scale)

        # calculate texture loss
        texture_loss = (self.calcu_texture_loss(fusion_images, ir_images, vis_images)
                        * self.texture_scale)

        # calculate intensity loss
        intensity_loss = (self.calcu_intensity_loss(fusion_images,
                                                    ir_images,
                                                    vis_images)
                          * self.intensity_scale)

        if self.use_psnr:
            psnr_loss = (self.calcu_psnr_loss(fusion_images, ir_images, vis_images)
                         * self.psnr_scale)
        else:
            psnr_loss = torch.tensor(0.0)

        # calculate total loss
        total_loss = (
                the_ssim_loss * self.ssim_loss_ratio
                + texture_loss * self.texture_loss_ratio
                + intensity_loss * self.intensity_loss_ratio
                + psnr_loss * self.psnr_loss_ratio
        )

        key_list = ["ssim_loss", "texture_loss", "intensity_loss", "psnr_loss", "total_loss"]
        value_list = [the_ssim_loss, texture_loss, intensity_loss, psnr_loss, total_loss]
        # 从tensor转为普通数字，round 5
        for i, elem in enumerate(value_list):
            value_list[i] = round(elem.item(), 5)
        # 构造详细loss信息的字典
        loss_state_dict = dict(zip(key_list, value_list))

        # 记录详细信息
        self.loss_recorder_in_detail.record(loss_state_dict)
        # 返回tensor用于计算backward()，和详细信息，两项
        return total_loss, loss_state_dict

    def calcu_history_mean_and_clear_and_save_to_mean_recorder(self) -> dict:
        """
        Calculates the mean of the loss history and clears the loss history.
        注意所有self.record_stack中是多个字典。其中字典的value值都是python的float而不是tensor
        Returns:
            The mean of the loss history
        """
        # 由dict的list创建一个values的list，每项是4个value。每项都是一个dict_value类的对象，是可迭代的，当做list看待
        record_stack_dict_values_list = [dic.values() for dic in self.loss_recorder_in_detail.record_stack]
        # 改变组织结构，每个value分别形成一个单独的tuple，总共四个tuple
        tuples = zip(*record_stack_dict_values_list)
        arrays = [np.array(tuple_) for tuple_ in tuples]
        means = [round(np.mean(array), 5) for array in arrays]

        self.loss_recorder_in_detail.delete_all()

        key_list_for_means_dict = [
            "ssim_loss_mean",
            "texture_loss_mean",
            "intensity_loss_mean",
            "psnr_loss_mean",
            "total_loss_mean",
        ]

        means_dict = dict(zip(key_list_for_means_dict, means))
        self.mean_loss_recorder.record(means_dict)
        return means_dict


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
