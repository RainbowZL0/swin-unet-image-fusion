import os

import colorama
import cv2
import torch
from colorama import Fore
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import A000_CONFIG as CFG
from a013_ModelDefinition import MyModel
from a015_dataset import MyDataset
from a016_train import get_time_str


class MyTest:
    def __init__(self):
        self.model = MyModel(
            window_size=CFG.WINDOW_SIZE,
            merging_size=CFG.MERGING_SIZE,
            in_dims_list=CFG.IN_DIMS_LIST,
            out_dims_list=CFG.OUT_DIMS_LIST,
            att_num_heads=CFG.ATT_NUM_HEADS,
            att_dims_per_head_ratio=CFG.ATT_DIMS_PER_HEAD_RATIO,
            attention_drop_ratio=CFG.ATTENTION_DROP_RATIO,
            linear_after_att_drop_ratio=CFG.LINEAR_AFTER_ATT_DROP_RATIO,
            mlp_hidden_dims_ratio=CFG.MLP_HIDDEN_DIMS_RATIO,
            mlp_activation_func=CFG.MLP_ACTIVATION_FUNC,
            mlp_drop_ratio=CFG.MLP_DROP_RATIO,
            final_layer_att_dims_per_head_ratio=CFG.FINAL_LAYER_ATT_DIMS_PER_HEAD_RATIO,
            final_conv_layer_kernel_size=CFG.FINAL_CONV_LAYER_KERNEL_SIZE,
            final_layer_mlp_hidden_dims_ratio=CFG.FINAL_LAYER_MLP_HIDDEN_DIMS_RATIO,
        ).to(device=CFG.DEVICE)

        self.test_dts = MyDataset(
            is_test=True,
            dataset_folder=CFG.TEST_DATASET_FOLDER,
        )
        self.test_dtl = DataLoader(
            dataset=self.test_dts,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        colorama.init(autoreset=True)

    def load_my_state(self):
        print(Fore.CYAN + f"Loading state from '{CFG.USING_STATE_PATH}'")
        read_state = torch.load(CFG.USING_STATE_PATH, map_location=CFG.DEVICE)
        self.model.load_state_dict(read_state["model_state"])
        print(Fore.CYAN + f"State is loaded successfully.")

    def test_fusion(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch_dict in tqdm(enumerate(self.test_dtl, start=1), initial=1):
                batch_dict: dict
                ir, vis, ir_path_list, vis_path_list = batch_dict.values()
                ir: Tensor
                vis: Tensor

                # 当Dataset的is_test参数传入为True时，vis有三个通道YCrCb，前向传播只需要Y通道。然后将输出的Y通道与原始的CrCb合并
                # 当Dataset的is_test为False时，vis只有一个通道Y，因为训练的时候只需要Y
                # shape "b c h w"
                vis_y, fus_cr_cb = vis[:, 0:1, :, :], vis[:, 1:3, :, :]

                ir = ir.to(device=CFG.DEVICE)
                vis_y = vis_y.to(device=CFG.DEVICE)
                fus_y = self.model(ir, vis_y)

                # # 输出结果先min max norm到[0, 1]，再用原始的数据scale back
                # fus_y = scale_to_interval_01(fus_y)
                # fus_y = scale_back_from_min_max_norm_01(
                #     fus_y,
                #     torch.min(vis_y),
                #     torch.max(vis_y)
                # )

                fus_y = fus_y.detach().cpu()
                fus_y = torch.clamp_(input=fus_y, min=0, max=1)
                fus_y_cr_cb = torch.concat([fus_y, fus_cr_cb], dim=1).squeeze(0)  # to shape "c h w", c=3

                fus_y_cr_cb = fus_y_cr_cb.permute(1, 2, 0).numpy()
                fus_y_cr_cb = cv2.cvtColor(src=fus_y_cr_cb, code=cv2.COLOR_YCrCb2RGB)
                fus_y_cr_cb = torch.from_numpy(fus_y_cr_cb).permute(2, 0, 1)

                save_test_result(fus_y_cr_cb, ir_path_list[0])


def scale_back_from_min_max_norm_01(x, original_min, original_max):
    """假设输入是经过min_max_norm的，反向恢复出未经过norm的"""
    x *= original_max - original_min
    x += original_min
    return x


def save_test_result(fus, ir_path):
    """
    Args:
        fus: Tensor with shape "c h w", c=3
        ir_path: original ir path
    Returns:
        Nothing
    """
    time_str = get_time_str()
    original_name_with_suffix = os.path.basename(ir_path)
    original_name = os.path.splitext(original_name_with_suffix)[0]
    # file_name = f"{time_str}_{original_name}.jpg"
    file_name = f"{original_name}_MKX_SELF.jpg"
    save_to_path = os.path.join(CFG.TEST_RESULT_FOLDER, file_name)
    save_image(tensor=fus, fp=save_to_path)
    print(Fore.GREEN + f"Result of {original_name_with_suffix} saved to {save_to_path}")


def start_test():
    my_test_obj = MyTest()
    my_test_obj.load_my_state()
    my_test_obj.test_fusion()


if __name__ == '__main__':
    start_test()
