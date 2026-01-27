from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Optional, Tuple, List


# Decoder: p(x | z_d, z_x, z_y)
class PX(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super().__init__()
        self.n_feature = args.n_feature

        self.fc1 = nn.Sequential(
            nn.Linear(zd_dim + zx_dim + zy_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.un1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU(),
        )

        self.un2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=512, kernel_size=(1, 1)),
            nn.ReLU(),
        )

        self.un3 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=(1, 1)),
            nn.ReLU(),
        )

        self.un4 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=self.n_feature, kernel_size=(1, 5)),
            nn.ReLU(),
        )

        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.deconv1[0].weight)
        nn.init.xavier_uniform_(self.deconv2[0].weight)
        nn.init.xavier_uniform_(self.deconv3[0].weight)
        nn.init.xavier_uniform_(self.deconv4[0].weight)

    def forward(self, zd, zx, zy, idxs, sizes):
        if zx is None:
            z = torch.cat((zd, zy), dim=-1)
        else:
            z = torch.cat((zd, zx, zy), dim=-1)

        h = self.fc1(z)
        h = h.view(-1, 64, 1, 8)

        out_1 = self.un1(h, idxs[3], output_size=sizes[2])
        out_11 = self.deconv1(out_1)

        out_2 = self.un2(out_11, idxs[2], output_size=sizes[1])
        out_22 = self.deconv2(out_2)

        out_3 = self.un3(out_22, idxs[1], output_size=sizes[0])
        out_33 = self.deconv3(out_3)

        out_4 = self.un4(out_33, idxs[0])
        out_44 = self.deconv4(out_4)

        out = out_44.permute(0, 3, 1, 2)
        return out


# Priors: p(z_d | d), p(z_y | y)
class DomainPrior(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super().__init__()
        self.d_dim = d_dim
        self.device = args.device

        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.eye(self.d_dim, device=self.device)
        d_onehot = a[d]
        hidden = self.fc1(d_onehot)
        mu = self.fc21(hidden)
        scale = self.fc22(hidden) + 1e-7
        return mu, scale


class ClassPrior(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super().__init__()
        self.y_dim = y_dim
        self.device = args.device

        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.eye(self.y_dim, device=self.device)
        y_onehot = a[y]
        hidden = self.fc1(y_onehot)
        mu = self.fc21(hidden)
        scale = self.fc22(hidden) + 1e-7
        return mu, scale


# Encoders: q(z_d|x), q(z_y|x)
class QZD(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super().__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_feature, 1024, kernel_size=(1, 5)), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1)), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(512, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(512, zd_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.conv1[0].weight)
        nn.init.xavier_uniform_(self.conv2[0].weight)
        nn.init.xavier_uniform_(self.conv3[0].weight)
        nn.init.xavier_uniform_(self.conv4[0].weight)

        nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = x.float()
        x_img = x.view(-1, x.shape[2], 1, x.shape[1])  # (B, F, 1, T)

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])  # (B, 64*8)=512
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        mu = self.fc11(out)
        scale = self.fc12(out) + 1e-7

        return mu, scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


class QZY(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim, args):
        super().__init__()
        self.n_feature = args.n_feature

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_feature, 1024, kernel_size=(1, 5)), nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)), nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1)), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Sequential(nn.Linear(512, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(512, zy_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.conv1[0].weight)
        nn.init.xavier_uniform_(self.conv2[0].weight)
        nn.init.xavier_uniform_(self.conv3[0].weight)
        nn.init.xavier_uniform_(self.conv4[0].weight)

        nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = x.float()
        x_img = x.view(-1, x.shape[2], 1, x.shape[1])  # (B, F, 1, T)

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])  # 512
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        mu = self.fc11(out)
        scale = self.fc12(out) + 1e-7

        return mu, scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


# Auxiliary heads: q(d|z_d), q(y|z_y)
class QD(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = F.relu(zd)
        return self.fc1(h)


class QY(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()
        self.fc1 = nn.Linear(zy_dim, y_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = F.relu(zy)
        return self.fc1(h)


# Main model: GILE(args)
class GILE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.zd_dim = args.d_AE
        self.zx_dim = 0
        self.zy_dim = args.d_AE
        self.d_dim = args.n_domains
        self.x_dim = args.x_dim
        self.y_dim = args.n_class

        self.px = PX(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzd = DomainPrior(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.pzy = ClassPrior(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qzd = QZD(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)
        self.qzy = QZY(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim, args)

        self.qd = QD(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = QY(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        self.to(args.device)

    def forward(self, d, x, y):
        # Encode
        zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)

        # Reparameterization
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, None, zy_q, idxs_y, sizes_y)

        # Priors
        zd_p_loc, zd_p_scale = self.pzd(d)
        zy_p_loc, zy_p_scale = self.pzy(y)

        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary logits
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        qzx = None
        pzx = None
        zx_q = None
        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y):
        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        CE_x = F.mse_loss(x_recon, x.float())

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        KL_zx = 0 
        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

        CE_d = F.cross_entropy(d_hat, d, reduction="sum")
        CE_y = F.cross_entropy(y_hat, y, reduction="sum")

        total = (
            CE_x
            - self.beta_d * zd_p_minus_zd_q
            - self.beta_x * KL_zx
            - self.beta_y * zy_p_minus_zy_q
            + self.aux_loss_multiplier_d * CE_d
            + self.aux_loss_multiplier_y * CE_y
        )

        return total, CE_y

    def loss_function_false(self, args, d, x, y):
        pred_d, pred_y, pred_d_false, pred_y_false = self.classifier(x)

        loss_classify_true = args.weight_true * (
            F.cross_entropy(pred_d, d, reduction="sum") + F.cross_entropy(pred_y, y, reduction="sum")
        )
        loss_classify_false = args.weight_false * (
            F.cross_entropy(pred_d_false, d, reduction="sum") + F.cross_entropy(pred_y_false, y, reduction="sum")
        )

        loss = loss_classify_true - loss_classify_false
        loss.requires_grad = True
        return loss

    def classifier(self, x):
        with torch.no_grad():
            zd_q_loc, _, _, _ = self.qzd(x)
            zd = zd_q_loc
            alpha_d = F.softmax(self.qd(zd), dim=1)
            _, ind = torch.topk(alpha_d, 1)
            d = x.new_zeros(alpha_d.size()).scatter_(1, ind, 1.0)

            zy_q_loc, _, _, _ = self.qzy(x)
            zy = zy_q_loc
            alpha_y = F.softmax(self.qy(zy), dim=1)
            _, ind = torch.topk(alpha_y, 1)
            y = x.new_zeros(alpha_y.size()).scatter_(1, ind, 1.0)

            alpha_y2d = F.softmax(self.qd(zy), dim=1)
            _, ind = torch.topk(alpha_y2d, 1)
            d_false = x.new_zeros(alpha_y2d.size()).scatter_(1, ind, 1.0)

            alpha_d2y = F.softmax(self.qy(zd), dim=1)
            _, ind = torch.topk(alpha_d2y, 1)
            y_false = x.new_zeros(alpha_d2y.size()).scatter_(1, ind, 1.0)

        return d, y, d_false, y_false

    def get_features(self, x):
        zy_q_loc, zy_q_scale, _, _ = self.qzy(x)
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        return qzy.rsample()
from argparse import Namespace
import torch

def make_gile_args_for_dataset(
    *,
    n_feature: int,      # F
    n_class: int,        # #activities
    n_domains: int,      # #domains
    device: torch.device,
    d_AE: int = 50,
    x_dim: int = 1152,   
    aux_loss_multiplier_y: float = 1000.0,
    aux_loss_multiplier_d: float = 1000.0,
    beta_d: float = 1.0,
    beta_x: float = 0.0,
    beta_y: float = 1.0,
    weight_true: float = 1000.0,
    weight_false: float = 1000.0,
):
    return Namespace(
        device=device,
        n_feature=n_feature,
        n_class=n_class,
        n_domains=n_domains,
        d_AE=d_AE,
        x_dim=x_dim,
        aux_loss_multiplier_y=aux_loss_multiplier_y,
        aux_loss_multiplier_d=aux_loss_multiplier_d,
        beta_d=beta_d,
        beta_x=beta_x,
        beta_y=beta_y,
        weight_true=weight_true,
        weight_false=weight_false,
    )
