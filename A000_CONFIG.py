import os

import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_SAVED_STATE = False
USING_STATE_PATH = r".\a002_saved_history\a001_models\04.20.01.23_epoch20.pth"
# USING_STATE_PATH = r".\a002_saved_history\a001_models\04.20.01.23_epoch20.pth"

SAVE_MODEL_TO_FOLDER = r".\a002_saved_history\a001_models"
SAVE_VALI_RESULTS_TO_FOLDER = r".\a002_saved_history\a002_vali_results"

TRAINING_DATASET_FOLDER = r"D:\_MyProjects\_Python\Grade4FirstHalf\a001_DL\a009_VIF\_DATASET\b000_MKX\train"
TEST_DATASET_FOLDER = r".\test\input"
TEST_RESULT_FOLDER = r".\test\output"

LR = 1e-2
MINIMUM_LR = 1e-5
SCHEDULER_T0 = 20
EPOCH = 20

BATCH_SIZE = 20
TRAINING_SET_RATIO = 0.99
DROP_LAST = True

PRINT_TRAINING_INFO_IN_INTERS = 5
VALI_INTERVAL_IN_ITERS = 100
SAVE_MODEL_INTERVAL_IN_EPOCHS = 1

"""loss相关"""
# 是否使用multi scale的ssim
CHOOSE_MS_SSIM = True
FUS_IR_SSIM_WEIGHT = 0.2  # ssim中，fusion和ir的相似度的权重; 1减去该值为fusion和vis的相似度权重
# 纹理边缘loss, Canny or Sobel
CHOOSE_CANNY_ELSE_SOBEL = False
# PSNR相关
USE_PSNR = False
FUS_IR_PSNR_WEIGHT = 0.4

# 放缩loss的倍数，使得他们有接近的数量级。其实和下面的merging weight的作用类似，只是有不同的含义。
# 日志输出的loss是经过scale的
SSIM_SCALE = 0.305
TEXTURE_SCALE = 250
INTENSITY_SCALE = 45
PSNR_SCALE = 0
# 假设loss已经有相同的数量级，加权平均成为总loss。merging weight to produce total weight, important
SSIM_LOSS_RATIO = 1/3
TEXTURE_LOSS_RATIO = 1/3
INTENSITY_LOSS_RATIO = 1/3
PSNR_LOSS_RATIO = 0

"""model构造参数"""
WINDOW_SIZE = (7, 7)
MERGING_SIZE = (2, 2)
IN_DIMS_LIST = [1, 24, 48, 96, 192]
OUT_DIMS_LIST = [24, 48, 96, 192, 384]
ATT_NUM_HEADS = 8
ATT_DIMS_PER_HEAD_RATIO = 1 / 8
ATTENTION_DROP_RATIO = 0
LINEAR_AFTER_ATT_DROP_RATIO = 0
MLP_HIDDEN_DIMS_RATIO = 4
MLP_ACTIVATION_FUNC = nn.ELU(inplace=True)
MLP_DROP_RATIO = 0

FINAL_LAYER_ATT_DIMS_PER_HEAD_RATIO = 1
FINAL_LAYER_MLP_HIDDEN_DIMS_RATIO = 1
FINAL_CONV_LAYER_KERNEL_SIZE = 3

"""log相关"""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
TENSOR_BOARD_LOG_DIR = None
TENSOR_BOARD_FLUSH_INTERVAL_SECS = 60


"""epsilon"""
EPSILON = 1e-10  # very small positive value

if __name__ == '__main__':
    pass
