import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from gvsl_models.gvsl import GVSL
from _gvsl_utils.STN import SpatialTransformer, AffineTransformer
from _gvsl_utils.Transform_self import SpatialTransform, AppearanceTransform
from _gvsl_utils.dataloader_heart_self_train import (
    DatasetFromFolder3D as DatasetFromFolder3D_train,
)
from _gvsl_utils.losses import gradient_loss, ncc_loss, MSE
from _gvsl_utils.gvsl_utils import AverageMeter
import numpy as np
from utils import config, get_pretrain_datalist
from pathlib import Path


class Trainer(object):
    def __init__(
        self,
        k=0,
        lr=1e-4,
        epoches=1000,
        iters=200,
        batch_size=1,
        model_name="GVSL",
        # unlabeled_dir="",
        datalist=[],
        results_dir="results",
        checkpoint_dir=Path(config.tensorboard_dir) / "pretrain" / "pretrain_GVSL",
    ):
        super(Trainer, self).__init__()
        self.k = k
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        # self.unlabeled_dir = unlabeled_dir
        self.datalist = datalist

        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # Data augmentation
        self.spatial_aug = SpatialTransform(
            do_rotation=True,
            angle_x=(-np.pi / 9, np.pi / 9),
            angle_y=(-np.pi / 9, np.pi / 9),
            angle_z=(-np.pi / 9, np.pi / 9),
            do_scale=True,
            scale_x=(0.75, 1.25),
            scale_y=(0.75, 1.25),
            scale_z=(0.75, 1.25),
            do_translate=True,
            trans_x=(-0.1, 0.1),
            trans_y=(-0.1, 0.1),
            trans_z=(-0.1, 0.1),
            do_shear=True,
            shear_xy=(-np.pi / 18, np.pi / 18),
            shear_xz=(-np.pi / 18, np.pi / 18),
            shear_yx=(-np.pi / 18, np.pi / 18),
            shear_yz=(-np.pi / 18, np.pi / 18),
            shear_zx=(-np.pi / 18, np.pi / 18),
            shear_zy=(-np.pi / 18, np.pi / 18),
            do_elastic_deform=True,
            alpha=(0.0, 512.0),
            sigma=(10.0, 13.0),
        )

        # transformation for restoration
        self.style_aug = AppearanceTransform(
            local_rate=0.8, nonlinear_rate=0.9, paint_rate=0.9, inpaint_rate=0.2
        )

        # initialize model
        self.gvsl = GVSL()

        if torch.cuda.is_available():
            self.gvsl = self.gvsl.cuda()

        self.opt = torch.optim.Adam(self.gvsl.parameters(), lr=lr)

        self.stn = SpatialTransformer()
        self.atn = AffineTransformer()

        self.softmax = nn.Softmax(dim=1)

        # initialize the dataloader
        # train_dataset = DatasetFromFolder3D_train(self.unlabeled_dir)
        train_dataset = DatasetFromFolder3D_train(self.datalist)
        self.dataloader_train = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # define loss
        self.L_smooth = gradient_loss
        self.L_ncc = ncc_loss
        self.L_mse = MSE

        # define loss log
        self.L_ncc_log = AverageMeter(name="L_ncc")
        self.L_MSE_log = AverageMeter(name="L_MSE")
        self.L_smooth_log = AverageMeter(name="L_smooth")

    def train_iterator(self, unlabed_img1, unlabed_img1_aug, unlabed_img2):

        res_A, warp_BA, aff_mat_BA, flow_BA = self.gvsl(unlabed_img1_aug, unlabed_img2)
        loss_ncc = self.L_ncc(warp_BA, unlabed_img1)
        self.L_ncc_log.update(loss_ncc.data, unlabed_img1.size(0))

        loss_mse = self.L_mse(res_A, unlabed_img1)
        self.L_MSE_log.update(loss_mse.data, unlabed_img1.size(0))

        loss_smooth = self.L_smooth(flow_BA)
        self.L_smooth_log.update(loss_smooth.data, unlabed_img1.size(0))

        loss = loss_ncc + loss_mse + loss_smooth

        loss.backward()
        self.opt.step()
        self.gvsl.zero_grad()
        self.opt.zero_grad()

    def train_epoch(self, epoch):
        self.gvsl.train()
        for i in range(self.iters):
            unlabed_img1, unlabed_img2 = next(self.dataloader_train.__iter__())

            # damage the image 1 for restoration
            unlabed_img1_aug = unlabed_img1.data.numpy()[0].copy()
            unlabed_img1_aug = self.style_aug.rand_aug(unlabed_img1_aug)
            unlabed_img1_aug = torch.from_numpy(
                unlabed_img1_aug[np.newaxis, :, :, :, :]
            )

            if torch.cuda.is_available():
                unlabed_img1 = unlabed_img1.cuda()
                unlabed_img2 = unlabed_img2.cuda()
                unlabed_img1_aug = unlabed_img1_aug.cuda()

            # Augment the image 1, damaged image 1 and image 2
            mat, code_spa = self.spatial_aug.rand_coords(unlabed_img1.shape[2:])
            unlabed_img1_aug = self.spatial_aug.augment_spatial(
                unlabed_img1_aug, mat, code_spa
            )
            unlabed_img1 = self.spatial_aug.augment_spatial(unlabed_img1, mat, code_spa)
            unlabed_img2 = self.spatial_aug.augment_spatial(unlabed_img2, mat, code_spa)

            self.train_iterator(unlabed_img1, unlabed_img1_aug, unlabed_img2)
            res = "\t".join(
                [
                    "Epoch: [%d/%d]" % (epoch + 1, self.epoches),
                    "Iter: [%d/%d]" % (i + 1, self.iters),
                    self.L_ncc_log.__str__(),
                    self.L_smooth_log.__str__(),
                    self.L_MSE_log.__str__(),
                ]
            )
            print(res)

    def checkpoint(self, epoch):
        torch.save(
            self.gvsl.state_dict(),
            "{0}/{1}_epoch_{2}.pth".format(
                self.checkpoint_dir, self.model_name, epoch + self.k
            ),
        )

    def load(self):
        self.gvsl.load_state_dict(
            torch.load(
                "{0}/{1}_epoch_{2}.pth".format(
                    self.checkpoint_dir, self.model_name, str(self.k)
                )
            )
        )

    def train(self):
        for epoch in range(self.epoches - self.k):
            self.L_ncc_log.reset()
            self.L_MSE_log.reset()
            self.L_smooth_log.reset()
            self.train_epoch(epoch + self.k)
            if epoch % 20 == 0:
                self.checkpoint(epoch)
        self.checkpoint(self.epoches - self.k)


if __name__ == "__main__":
    datalist = get_pretrain_datalist(img_list=True)
    trainer = Trainer(datalist=datalist)
    # trainer.load()
    trainer.train()
