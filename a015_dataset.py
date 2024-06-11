import os
import torch
from typing import Tuple, List

import numpy as np
from torchvision.transforms import v2
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tqdm import tqdm
from textwrap import dedent


class MyDataset(Dataset):
    def __init__(self, is_test, dataset_folder):
        """
        init的目标是获取vis和ir图片的路径，形成两个list，作为类的属性。
        访问数据时会自动调用__getitem__()方法，返回字典
        Args:
            is_test: True为训练集或验证集。False为测试集。区别是vis返回的通道数量，具体看__getitem__()
            dataset_folder: 数据
        """
        # copy args to attributes
        self.is_test = is_test
        self.dataset_folder = Path(dataset_folder)

        """new attributes"""
        # 数据集
        self.ir_path_list, self.vis_path_list = self.get_ir_vis_paths()
        # 数据增强
        self.np_generator = np.random.default_rng()  # 随机数生成器的构造函数，返回一种现代的numpy的生成器的类
        self.compose_stage_1 = self.get_compose_stage_1()  # transform
        # 保存64位全1的64位最大整数，用于指定生成的随机数的范围，用作随机种子。这个数越大，随机状态数量越多。
        self.max_int = 1 << 64 - 1
        # 数据标准化
        # TODO

    def get_ir_vis_paths(self) -> Tuple[List, List]:
        """
        Returns: 按照文件名排序之后的两个list
        """
        ir_path_list = []
        vis_path_list = []
        for root, dirs, files in os.walk(self.dataset_folder):
            root_basename = os.path.basename(root)  # root这个文件夹路径的最后一级的名字
            if root_basename == "ir":
                collect_path(collective_list=ir_path_list, root=root, file_names_list=files)
            elif root_basename == "vis":
                collect_path(collective_list=vis_path_list, root=root, file_names_list=files)
        return sorted(ir_path_list), sorted(vis_path_list)

    def get_compose_stage_1(self):
        """
        注意Transform类的容器可以是Compose或Sequential。构造前者时需要一个列表，类似ModuleList
        Returns: 一个Compose类，按顺序装有所需的数据增强。
        """
        transform_list = [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ]
        if not self.is_test:
            transform_list.extend([
                v2.RandomResizedCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        return v2.Compose(transform_list)

    def __len__(self):
        return len(self.ir_path_list)

    def __getitem__(self, index) -> dict:
        ir_path, vis_path = self.ir_path_list[index], self.vis_path_list[index]
        ir, vis = (cv2.imread(filename=ir_path, flags=cv2.IMREAD_GRAYSCALE),
                   cv2.imread(filename=vis_path, flags=cv2.IMREAD_COLOR))

        # 如果路径错误，cv2.imread会返回None，避免这种情况，做检查
        if ir is None or vis is None:
            raise NameError(dedent(
                f"""
                either ir or vis cannot be None, but got None, at position:
                ir_path = {ir_path},
                vis_path = {vis_path}
                """
            ))

        # 确认二者都不为None之后，预处理。
        # ir加一个维度为"h w 1"
        # vis需要转换颜色为YCrCb
        ir = ir[..., np.newaxis]
        vis = cv2.cvtColor(src=vis, code=cv2.COLOR_BGR2YCrCb)
        # 如果是train或vali，vis只返回Y通道。否则为test，返回YCrCb三个通道
        if not self.is_test:
            vis = vis[..., 0:1]

        # 同样的条件判断，但是目的不同，为了数据增强
        if not self.is_test:
            my_seed = self.np_generator.integers(low=0, high=self.max_int, size=None, dtype=np.int64)

            # 使用相同的seed处理这一对ir和vis
            torch.manual_seed(my_seed)
            ir = self.compose_stage_1(ir)
            torch.manual_seed(my_seed)
            vis = self.compose_stage_1(vis)
        else:
            # 测试集，没有数据增强，只是转换ndarray为tensor，并且转换数值类型从int 0~255到float 0~1
            ir, vis = self.compose_stage_1(ir), self.compose_stage_1(vis)

        return {
            "ir": ir,
            "vis": vis,
            "ir_path": ir_path,
            "vis_path": vis_path,
        }

    def get_item_(self, index) -> dict:
        """
        避免使用中括号索引，因为没有类型提示。
        Args:
            index: 索引
        Returns: 指定索引的数据
        """
        return self.__getitem__(index)


def collect_path(collective_list: list, root: str, file_names_list: list):
    """
    收集文件路径到一个完整的list中
    Args:
        collective_list: 完整的list
        root: 文件所在的文件夹
        file_names_list: list，包含所有文件名
    Returns:
        更新后的collective_list
    """
    for file_name in file_names_list:
        file_abs_path = os.path.join(root, file_name)
        collective_list.append(file_abs_path)
    return collective_list


def test_my_dataset():
    dataset_folder = r"D:\_MyProjects\_Python\Grade4FirstHalf\a001_DL\a009_VIF\_DATASET\b000_MKX\train"
    my_dataset = MyDataset(is_test=True, dataset_folder=dataset_folder)
    for i in tqdm(range(len(my_dataset))):
        my_dataset.get_item_(i).values()


if __name__ == '__main__':
    test_my_dataset()
