import os
from datetime import datetime
from textwrap import dedent

import colorama
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import nn, optim, Tensor
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import A000_CONFIG as CFG
from a008_loss import MyLoss
from a013_ModelDefinition import MyModel
from a015_dataset import MyDataset


class MyTraining:
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
        self.model.apply(init_params)

        # 数据集相关
        self.dataset = MyDataset(is_test=False, dataset_folder=CFG.TRAINING_DATASET_FOLDER)
        self.tr_set, self.val_set = random_split(
            dataset=self.dataset,
            lengths=[CFG.TRAINING_SET_RATIO, 1 - CFG.TRAINING_SET_RATIO]
        )
        self.tr_dtl = DataLoader(
            dataset=self.tr_set,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            drop_last=CFG.DROP_LAST,
            num_workers=0,
        )
        self.val_dtl = DataLoader(
            dataset=self.val_set,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            drop_last=CFG.DROP_LAST,
            num_workers=0,
        )
        self.len_tr_dtl = len(self.tr_dtl)

        # optimizer, scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=CFG.LR)
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer,
            T_0=CFG.SCHEDULER_T0,
            eta_min=CFG.MINIMUM_LR,
        )

        # 计算loss
        self.tr_loss_calculator = MyLoss().to(device=CFG.DEVICE)
        self.val_loss_calculator = MyLoss().to(device=CFG.DEVICE)

        """日志相关"""
        # 记录当前epoch和iter数量，便于输出日志
        self.current_epochs = 1
        self.current_iters_in_one_epoch = 1
        self.current_total_iters = self.calcu_current_total_iters()

        colorama.init(autoreset=True)

        self.tensorboard_writer = SummaryWriter(
            log_dir=CFG.TENSOR_BOARD_LOG_DIR,
            flush_secs=CFG.TENSOR_BOARD_FLUSH_INTERVAL_SECS,
        )

    def start_train(self):
        epoch_range = range(self.current_epochs, CFG.EPOCH + 1, 1)
        print(
            Fore.CYAN +
            f"Starting at epoch = {self.current_epochs}, "
            f"using lr = {self.scheduler.get_last_lr()[0]}, "
            f"iters in one epoch = {self.len_tr_dtl}"
        )

        for epoch in tqdm(iterable=epoch_range, initial=1):
            self.current_epochs = epoch
            self.train_one_epoch_with_vali()

            if epoch % CFG.SAVE_MODEL_INTERVAL_IN_EPOCHS == 0:
                self.save_my_state()

        self.tensorboard_writer.close()

    def use_scheduler(self):
        current_step = self.current_epochs-1 + (self.current_iters_in_one_epoch-1) / len(self.tr_dtl)
        # noinspection PyTypeChecker
        self.scheduler.step(current_step)
        print(Fore.GREEN + f"Learning rate changed to {self.scheduler.get_last_lr()[0]}")

    def calcu_current_total_iters(self):
        return ((self.current_epochs - 1) * len(self.tr_dtl)
                + self.current_iters_in_one_epoch)

    def submit_loss_to_tensorboard(self, detailed_loss_dict):
        for key, value in detailed_loss_dict.items():
            self.tensorboard_writer.add_scalar(
                tag=f"training/{key}",
                scalar_value=value,
                global_step=self.current_total_iters,
                new_style=True,
            )
        # 多加一项学习率用于debug，检查学习率是否按照预期变化
        self.tensorboard_writer.add_scalar(
            tag=f"training/lr",
            scalar_value=self.scheduler.get_last_lr()[0],
            global_step=self.current_total_iters,
            new_style=True,
        )
        self.tensorboard_writer.flush()

    def train_one_epoch_with_vali(self):
        self.model.train()
        for i, batch_dict in tqdm(iterable=enumerate(self.tr_dtl, start=1), initial=1):
            self.current_iters_in_one_epoch = i
            self.current_total_iters = self.calcu_current_total_iters()

            batch_dict: dict
            ir, vis, ir_path, vis_path = batch_dict.values()
            ir: Tensor = ir.to(device=CFG.DEVICE)
            vis: Tensor = vis.to(device=CFG.DEVICE)

            """实验尝试"""
            # ir, vis = (my_normalize(elem) for elem in (ir, vis))

            fusion = self.model(ir, vis)

            # ir, vis, fusion = (scale_to_interval_01(elem) for elem in (ir, vis, fusion))
            fusion = torch.clamp_(input=fusion, min=0, max=1)

            # detailed_loss_dict用于展示tensorboard
            loss, detailed_loss_dict = self.tr_loss_calculator.calcu_total_loss(
                fusion_images=fusion,
                ir_images=ir,
                vis_images=vis,
            )
            self.submit_loss_to_tensorboard(detailed_loss_dict)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.use_scheduler()

            # 输出训练loss
            if self.current_total_iters == 1 or self.current_total_iters % CFG.PRINT_TRAINING_INFO_IN_INTERS == 0:
                four_means_dict = self.tr_loss_calculator.calcu_history_mean_and_clear_and_save_to_mean_recorder()
                print(dedent(
                    f"""
                    Training Info: 
                    Epoch = {self.current_epochs},
                    Iter in this epoch = {self.current_iters_in_one_epoch},
                    Iter in total = {self.current_total_iters}
                    Loss = {four_means_dict}
                    """
                ))

            # 验证
            if self.current_total_iters % CFG.VALI_INTERVAL_IN_ITERS == 0:
                print(Fore.CYAN + f"Starting validation ...")
                self.vali()

                four_means_dict = self.val_loss_calculator.calcu_history_mean_and_clear_and_save_to_mean_recorder()
                print(Fore.CYAN + dedent(
                    f"""
                    Vali Info:
                    Epoch = {self.current_epochs},
                    Iter in this epoch = {self.current_iters_in_one_epoch},
                    Iter in total = {self.current_total_iters}
                    Loss = {four_means_dict}
                    """
                ))

    def vali(self):
        # 保存现场
        training_mode_record = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i, batch_dict in enumerate(self.val_dtl, start=1):
                batch_dict: dict
                ir, vis, ir_path, vis_path = batch_dict.values()
                ir: Tensor
                vis: Tensor
                ir = ir.to(device=CFG.DEVICE)
                vis = vis.to(device=CFG.DEVICE)

                """实验尝试"""
                # ir, vis = (my_normalize(elem) for elem in (ir, vis))

                fusion: Tensor = self.model(ir, vis)

                # ir, vis, fusion = (scale_to_interval_01(elem) for elem in (ir, vis, fusion))
                fusion = torch.clamp_(input=fusion, min=0, max=1)

                _, _ = self.val_loss_calculator.calcu_total_loss(
                    fusion_images=fusion,
                    ir_images=ir,
                    vis_images=vis,
                )

                # 在第一批次中，展示验证集上的融合结果。由于设定了vali_dataloader也是随机打乱的，所以每次采样都不一样。
                if i == 1:
                    self.save_vali_result(
                        ir=ir.detach().cpu(),
                        vis=vis.detach().cpu(),
                        fusion=fusion.detach().cpu()
                    )

        """恢复model的训练模式，恢复现场"""
        if training_mode_record:
            self.model.train()

    def save_my_state(self):
        time_str = get_time_str()
        model_file_name = f"{time_str}_epoch{self.current_epochs}.pth"
        save_state_to_path = os.path.join(CFG.SAVE_MODEL_TO_FOLDER, model_file_name)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "current_epoch": self.current_epochs,
        }
        torch.save(state, save_state_to_path)
        print(Fore.YELLOW + f"State saved to '{save_state_to_path}'")

    def save_vali_result(self, ir: Tensor, vis: Tensor, fusion: Tensor):
        """
        展示验证集上随机采样的batch_size个图片的融合结果
        Args:
            ir: 红外图片
            vis: 可见光图片的Y通道
            fusion: 融合图像
        """
        batch_size = ir.shape[0]

        image_pair_list = []
        for i in range(batch_size):
            image_pair = (
                ir[i, ...].permute(dims=(1, 2, 0)),  # "c h w" -> "h w c" 为了和matplotlib的imshow()兼容
                vis[i, ...].permute(dims=(1, 2, 0)),
                fusion[i, ...].permute(dims=(1, 2, 0))
            )

            image_pair_list.append(image_pair)

        fig = plt.figure()
        image_grid = ImageGrid(  # ImageGrid是一种axes的封装，需要指定一个所属的fig，第一个参数
            fig=fig,
            rect=111,  # 同一个fig上可以创建多个ImageGrid，通常只需要1个，这个参数是指定现在创建的ImageGrid是多少行，多少列，第几个
            nrows_ncols=(batch_size, 3),
            axes_pad=0,
            share_all=True,  # 共享两个轴，没有空隙
        )
        for i in range(batch_size):
            for j in range(3):
                k = i * 3 + j
                ax: Axes = image_grid[k]
                ax.imshow(image_pair_list[i][j], cmap="gray")
                ax.set_axis_off()
        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
            hspace=0,
            wspace=0,
        )

        time_str = get_time_str()
        saved_image_name = (f"{time_str}_"
                            f"Epoch{self.current_epochs}_"
                            f"EpIters{self.current_iters_in_one_epoch}_"
                            f"TotalIters{self.current_total_iters}.png")
        save_image_to_path = os.path.join(CFG.SAVE_VALI_RESULTS_TO_FOLDER, saved_image_name)
        fig.savefig(fname=save_image_to_path, bbox_inches="tight", pad_inches=0, dpi=1024)
        plt.close()

        print(Fore.CYAN + f"Validation results saved to '{save_image_to_path}'")

    def load_my_state(self):
        print(Fore.CYAN + f"Loading state from '{CFG.USING_STATE_PATH}'")

        read_state = torch.load(CFG.USING_STATE_PATH, map_location=CFG.DEVICE)
        #
        # read_model_state = read_state["model_state"]
        # read_optimizer_state = read_state["optimizer_state"]
        #
        # read_model_state, read_optimizer_state = self.del_params_in_model_state_and_optim_state_by_name(
        #     param_name_list=["final_merge_channel_layer.weight", "final_merge_channel_layer.bias"],
        #     model_state=read_model_state,
        #     optim_state=read_optimizer_state,
        # )
        #
        # now_model_state, now_optimizer_state = self.model.state_dict(), self.optimizer.state_dict()
        # now_model_state.update(read_model_state)
        # now_optimizer_state.update(read_optimizer_state)
        #
        # self.model.load_state_dict(now_model_state)
        # self.optimizer.load_state_dict(now_optimizer_state)

        # 分别load state
        self.model.load_state_dict(read_state["model_state"])
        self.optimizer.load_state_dict(read_state["optimizer_state"])
        self.scheduler.load_state_dict(read_state["scheduler_state"])
        self.current_epochs = read_state["current_epoch"] + 1  # current_epoch读取到的是已经结束的轮次，加一是下一轮
        #
        # self.tensorboard_writer = SummaryWriter(
        #     log_dir=MyConfig.TENSOR_BOARD_LOG_DIR,
        #     flush_secs=MyConfig.TENSOR_BOARD_FLUSH_INTERVAL_SECS,
        #     purge_step=self.calcu_current_total_iters(),
        # )

        print(Fore.CYAN + f"State is loaded successfully.")

    def del_params_in_model_state_and_optim_state_by_name(
            self,
            param_name_list: list,
            model_state: dict,
            optim_state: dict,
    ):
        for param_name in param_name_list:
            # 从模型状态中删除
            del model_state[param_name]
            # 从optimizer状态中删除
            params_id_list = list()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if name == param_name:
                    params_id_list.append(i)
            for id_ in params_id_list:
                del optim_state["state"][id_]
        return model_state, optim_state


def get_time_str():
    return datetime.now().strftime("%m.%d.%H.%M")


def my_normalize(x):
    return (x - torch.mean(x)) / (torch.std(x) + CFG.EPSILON)


def scale_to_interval_01(x):
    b, c, h, w = x.shape
    x = x.reshape(b, -1)

    min_val, _ = torch.min(x, dim=1, keepdim=True)
    max_val, _ = torch.max(x, dim=1, keepdim=True)

    x -= min_val
    x /= (max_val - min_val + CFG.EPSILON)

    x = x.reshape(b, c, h, w)
    return x


def init_params(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def start_main():
    training_obj = MyTraining()
    if CFG.USE_SAVED_STATE is True:
        training_obj.load_my_state()
    training_obj.start_train()


if __name__ == '__main__':
    start_main()
