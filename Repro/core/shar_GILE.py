from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Optional, List


# ---------------------------
# Decoder: p(x | z_d, z_x, z_y)  (SHAR-specific: 640 -> (64,1,10), kernel (1,4))
# ---------------------------
class PX(nn.Module):
    def __init__(self, n_feature: int, zd_dim: int, zx_dim: int, zy_dim: int):
        super().__init__()
        self.n_feature = n_feature

        self.fc1 = nn.Sequential(
            nn.Linear(zd_dim + zx_dim + zy_dim, 640, bias=False),
            nn.BatchNorm1d(640),
            nn.ReLU(),
        )

        self.un1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=(1, 1)), nn.ReLU())

        self.un2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 512, kernel_size=(1, 1)), nn.ReLU())

        self.un3 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(512, 1024, kernel_size=(1, 1)), nn.ReLU())

        self.un4 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, self.n_feature, kernel_size=(1, 4)),
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
        h = h.view(-1, 64, 1, 10)  # SHAR-specific

        out_1 = self.un1(h, idxs[3], output_size=sizes[2])
        out_11 = self.deconv1(out_1)

        out_2 = self.un2(out_11, idxs[2], output_size=sizes[1])
        out_22 = self.deconv2(out_2)

        out_3 = self.un3(out_22, idxs[1], output_size=sizes[0])
        out_33 = self.deconv3(out_3)

        out_4 = self.un4(out_33, idxs[0])
        out_44 = self.deconv4(out_4)

        # output layout: (B, 1, T, F)
        return out_44.permute(0, 2, 3, 1)


# ---------------------------
# Priors: p(z_d|d), p(z_y|y)
# ---------------------------
class DomainPrior(nn.Module):
    def __init__(self, d_dim: int, zd_dim: int, device: torch.device):
        super().__init__()
        self.d_dim = d_dim
        self.device = device

        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.ReLU())
        self.fc21 = nn.Linear(zd_dim, zd_dim)
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.zeros_(self.fc21.bias)
        nn.init.xavier_uniform_(self.fc22[0].weight)
        nn.init.zeros_(self.fc22[0].bias)

    def forward(self, d: torch.LongTensor):
        d = d.long().view(-1)
        d_onehot = F.one_hot(d, num_classes=self.d_dim).float().to(self.device)

        hidden = self.fc1(d_onehot)
        mu = self.fc21(hidden)
        scale = self.fc22(hidden) + 1e-7
        return mu, scale


class ClassPrior(nn.Module):
    def __init__(self, y_dim: int, zy_dim: int, device: torch.device):
        super().__init__()
        self.y_dim = y_dim
        self.device = device

        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.ReLU())
        self.fc21 = nn.Linear(zy_dim, zy_dim)
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.zeros_(self.fc21.bias)
        nn.init.xavier_uniform_(self.fc22[0].weight)
        nn.init.zeros_(self.fc22[0].bias)

    def forward(self, y: torch.LongTensor):
        y = y.long().view(-1)
        y_onehot = F.one_hot(y, num_classes=self.y_dim).float().to(self.device)

        hidden = self.fc1(y_onehot)
        mu = self.fc21(hidden)
        scale = self.fc22(hidden) + 1e-7
        return mu, scale


# ---------------------------
# Encoders: q(z_d|x), q(z_y|x)  (SHAR-specific conv kernel (1,4), flatten=640)
# expects x shape (B, T, F) and reshapes to (B, F, 1, T)
# ---------------------------
class QZD(nn.Module):
    def __init__(self, n_feature: int, zd_dim: int):
        super().__init__()
        self.n_feature = n_feature

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_feature, 1024, kernel_size=(1, 4)), nn.ReLU())
        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)), nn.ReLU())
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1)), nn.ReLU())
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.ReLU())
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Linear(640, zd_dim)
        self.fc12 = nn.Sequential(nn.Linear(640, zd_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.conv1[0].weight)
        nn.init.xavier_uniform_(self.conv2[0].weight)
        nn.init.xavier_uniform_(self.conv3[0].weight)
        nn.init.xavier_uniform_(self.conv4[0].weight)

        nn.init.xavier_uniform_(self.fc11.weight)
        nn.init.zeros_(self.fc11.bias)
        nn.init.xavier_uniform_(self.fc12[0].weight)
        nn.init.zeros_(self.fc12[0].bias)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> (B, F, 1, T)
        x_img = x.float().view(-1, x.shape[2], 1, x.shape[1])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])  # 64*10=640
        size1, size2, size3, size4 = out1.size(), out2.size(), out3.size(), out4.size()

        mu = self.fc11(out)
        scale = self.fc12(out) + 1e-7
        return mu, scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


class QZY(nn.Module):
    def __init__(self, n_feature: int, zy_dim: int):
        super().__init__()
        self.n_feature = n_feature

        self.conv1 = nn.Sequential(nn.Conv2d(self.n_feature, 1024, kernel_size=(1, 4)), nn.ReLU())
        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(1, 1)), nn.ReLU())
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=(1, 1)), nn.ReLU())
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.ReLU())
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        self.fc11 = nn.Linear(640, zy_dim)
        self.fc12 = nn.Sequential(nn.Linear(640, zy_dim), nn.Softplus())

        nn.init.xavier_uniform_(self.conv1[0].weight)
        nn.init.xavier_uniform_(self.conv2[0].weight)
        nn.init.xavier_uniform_(self.conv3[0].weight)
        nn.init.xavier_uniform_(self.conv4[0].weight)

        nn.init.xavier_uniform_(self.fc11.weight)
        nn.init.zeros_(self.fc11.bias)
        nn.init.xavier_uniform_(self.fc12[0].weight)
        nn.init.zeros_(self.fc12[0].bias)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> (B, F, 1, T)
        x_img = x.float().view(-1, x.shape[2], 1, x.shape[1])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])  # 640
        size1, size2, size3, size4 = out1.size(), out2.size(), out3.size(), out4.size()

        mu = self.fc11(out)
        scale = self.fc12(out) + 1e-7
        return mu, scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]


# ---------------------------
# Aux heads: q(d|z_d), q(y|z_y)
# ---------------------------
class QD(nn.Module):
    def __init__(self, zd_dim: int, d_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(zd_dim, d_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, zd):
        return self.fc1(F.relu(zd))


class QY(nn.Module):
    def __init__(self, zy_dim: int, y_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(zy_dim, y_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, zy):
        return self.fc1(F.relu(zy))


# ---------------------------
# Main model: GILE(self-config) (NO argparse/args Namespace)
# ---------------------------
class GILE(nn.Module):
    """
    SHAR GILE variant:
      - x input shape: (B, T, F)  (e.g., (B,151,3))
      - reconstruction shape: (B, 1, T, F)
      - conv kernel: (1,4)
      - flatten: 640
      - decoder seed: view(-1,64,1,10)

    You configure everything via __init__ params (like your older code style).
    """

    def __init__(
        self,
        n_feature: int = 3,          # F
        n_class: int = 17,           # activities
        n_domains: int = 6,          # domains
        d_AE: int = 50,              # latent dim for z_d and z_y
        x_dim: int = 1152,           # kept for compatibility (not used by conv)
        aux_loss_multiplier_y: float = 1000.0,
        aux_loss_multiplier_d: float = 1000.0,
        beta_d: float = 1.0,
        beta_x: float = 1.0,
        beta_y: float = 1.0,
        weight_true: float = 1000.0,
        weight_false: float = 1000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # dims
        self.n_feature = n_feature
        self.y_dim = n_class
        self.d_dim = n_domains
        self.zd_dim = d_AE
        self.zy_dim = d_AE
        self.zx_dim = 0
        self.x_dim = x_dim

        # weights
        self.aux_loss_multiplier_y = aux_loss_multiplier_y
        self.aux_loss_multiplier_d = aux_loss_multiplier_d
        self.beta_d = beta_d
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.weight_true = weight_true
        self.weight_false = weight_false

        # modules
        self.px = PX(self.n_feature, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = DomainPrior(self.d_dim, self.zd_dim, self.device)
        self.pzy = ClassPrior(self.y_dim, self.zy_dim, self.device)

        self.qzd = QZD(self.n_feature, self.zd_dim)
        self.qzy = QZY(self.n_feature, self.zy_dim)

        self.qd = QD(self.zd_dim, self.d_dim)
        self.qy = QY(self.zy_dim, self.y_dim)

        # move to device
        self.to(self.device)

    def forward(self, d, x, y):
        # x expected: (B, T, F)
        d = d.long().view(-1)
        y = y.long().view(-1)

        # Encode
        zd_q_loc, zd_q_scale, _, _ = self.qzd(x)
        zy_q_loc, zy_q_scale, idxs_y, sizes_y = self.qzy(x)

        # Reparam
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()

        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode -> (B, 1, T, F)
        x_recon = self.px(zd_q, None, zy_q, idxs_y, sizes_y)

        # Priors
        zd_p_loc, zd_p_scale = self.pzd(d)
        zy_p_loc, zy_p_scale = self.pzy(y)
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Aux logits
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        qzx = None
        pzx = None
        zx_q = None
        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y):
        d = d.long().view(-1)
        y = y.long().view(-1)

        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        # compare against (B,1,T,F)
        x_target = x.unsqueeze(1)

        CE_x = F.mse_loss(x_recon, x_target.float())

        # author-style "p - q" term using log_probs of sampled z
        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

        KL_zx = 0  # no z_x in this variant

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

    def loss_function_false(self, d, x, y):
        """
        IE / false step (author-style):
        note: classifier() is non-differentiable (no_grad + topk/scatter), matches author behavior.
        """
        d = d.long().view(-1)
        y = y.long().view(-1)

        pred_d, pred_y, pred_d_false, pred_y_false = self.classifier(x)

        loss_classify_true = self.weight_true * (
            F.cross_entropy(pred_d, d, reduction="sum") + F.cross_entropy(pred_y, y, reduction="sum")
        )
        loss_classify_false = self.weight_false * (
            F.cross_entropy(pred_d_false, d, reduction="sum") + F.cross_entropy(pred_y_false, y, reduction="sum")
        )

        loss = loss_classify_true - loss_classify_false
        loss.requires_grad = True
        return loss

    def classifier(self, x):
        # Author-style: no_grad + one-hot scatter
        with torch.no_grad():
            # d from zd
            zd_q_loc, _, _, _ = self.qzd(x)
            zd = zd_q_loc
            alpha_d = F.softmax(self.qd(zd), dim=1)
            _, ind = torch.topk(alpha_d, 1)
            d = x.new_zeros(alpha_d.size()).scatter_(1, ind, 1.0)

            # y from zy
            zy_q_loc, _, _, _ = self.qzy(x)
            zy = zy_q_loc
            alpha_y = F.softmax(self.qy(zy), dim=1)
            _, ind = torch.topk(alpha_y, 1)
            y = x.new_zeros(alpha_y.size()).scatter_(1, ind, 1.0)

            # false: d from zy
            alpha_y2d = F.softmax(self.qd(zy), dim=1)
            _, ind = torch.topk(alpha_y2d, 1)
            d_false = x.new_zeros(alpha_y2d.size()).scatter_(1, ind, 1.0)

            # false: y from zd
            alpha_d2y = F.softmax(self.qy(zd), dim=1)
            _, ind = torch.topk(alpha_d2y, 1)
            y_false = x.new_zeros(alpha_d2y.size()).scatter_(1, ind, 1.0)

        return d, y, d_false, y_false

    def get_features(self, x):
        zy_q_loc, zy_q_scale, _, _ = self.qzy(x)
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        return qzy.rsample()
