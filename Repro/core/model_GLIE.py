from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple
from utils import GradReverse
from typing import Tuple, List, Optional
import torch.distributions as dist

class DomainPrior(nn.Module):
    """
    p(z_d | d)
    Conditional Gaussian prior for domain latent variable z_d.
    """

    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()

        self.d_dim = d_dim
        self.latent_dim = zd_dim
        self.num_domains = 4
        self.backbone = nn.Sequential(
            nn.Linear(d_dim, zd_dim, bias=False),
            nn.BatchNorm1d(zd_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(zd_dim, zd_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(zd_dim, zd_dim),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.backbone[0].weight)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.xavier_uniform_(self.scale_head[0].weight)

        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.scale_head[0].bias)

    def forward(self, d: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            d: (B,) domain labels

        Returns:
            mu_d:    (B, latent_dim)
            sigma_d:(B, latent_dim)
        """
        d_onehot = F.one_hot(d, num_classes=self.num_domains).float()
        h = self.backbone(d_onehot)

        mu = self.mu_head(h)
        scale = self.scale_head(h) + 1e-7  # numerical stability

        return mu, scale
    
class ClassPrior(nn.Module):
    """
    p(z_y | y)
    Conditional Gaussian prior for class/activity latent variable z_y.
    """

    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()

        self.num_classes = y_dim
        self.latent_dim = zy_dim

        self.backbone = nn.Sequential(
            nn.Linear(y_dim, zy_dim, bias=False),
            nn.BatchNorm1d(zy_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(zy_dim, zy_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(zy_dim, zy_dim),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.backbone[0].weight)
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.xavier_uniform_(self.scale_head[0].weight)

        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.scale_head[0].bias)

    def forward(self, y: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            y: (B,) class labels

        Returns:
            mu_y:    (B, latent_dim)
            sigma_y:(B, latent_dim)
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        h = self.backbone(y_onehot)

        mu = self.mu_head(h)
        scale = self.scale_head(h) + 1e-7

        return mu, scale

class QZD(nn.Module):
    """
    q(z_d | x)
    Domain-specific latent encoder
    """

    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()

        self.n_features = 77

        # Convolutional backbone 
        self.conv1 = nn.Conv2d(self.n_features, 1024, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(512, 128, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        #zd_dim = num domains
        self.fc11 = nn.Sequential(nn.Linear(128, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(128, zd_dim), nn.Softplus())

        #  Latent heads 
        self.fc_mu = nn.Linear(128, zd_dim)
        self.fc_sigma = nn.Sequential(
            nn.Linear(128, zd_dim),
            nn.Softplus()
        )
        
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

        self._init_weights()


    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3])
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zd_loc = self.fc11(out)
        zd_scale = self.fc12(out) + 1e-7

        return zd_loc, zd_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]
    
class QZY(nn.Module):
    """
    q(z_y | x)
    Activity-specific latent encoder
    """

    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()
        self.n_features = 77

        # Convolutional backbone 
        self.conv1 = nn.Conv2d(self.n_features, 1024, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(512, 128, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        # zy_dim = d_AE
        self.fc11 = nn.Sequential(nn.Linear(128, 50))
        self.fc12 = nn.Sequential(nn.Linear(128, 50), nn.Softplus())
        # Latent heads 
        self.fc_mu = nn.Linear(128, zy_dim)
        self.fc_sigma = nn.Sequential(nn.Linear(128, zy_dim),
            nn.Softplus()
        )

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()


        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):

        x_img = x.float()
        x_img = x_img.view(-1, x_img.shape[3], 1, x_img.shape[2])

        out_conv1 = self.conv1(x_img)
        out1, idx1 = self.pool1(out_conv1)

        out_conv2 = self.conv2(out1)
        out2, idx2 = self.pool2(out_conv2)

        out_conv3 = self.conv3(out2)
        out3, idx3 = self.pool3(out_conv3)

        out_conv4 = self.conv4(out3)
        out4, idx4 = self.pool4(out_conv4)

        out = out4.reshape(-1, out4.shape[1] * out4.shape[3]) # [64, 512]
        size1 = out1.size()
        size2 = out2.size()
        size3 = out3.size()
        size4 = out4.size()

        zy_loc = self.fc11(out)
        zy_scale = self.fc12(out) + 1e-7

        return zy_loc, zy_scale, [idx1, idx2, idx3, idx4], [size1, size2, size3, size4]

class PX(nn.Module):
    """
    p(x | z_d, z_y [, z_x])

    Decoder using unpooling + transposed convolutions.
    """

    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()

        self.n_features = 77

        total_latent_dim = zd_dim + zx_dim + zy_dim

        #  Latent projection 
        self.fc = nn.Sequential(
            nn.Linear(total_latent_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        #  Unpool + deconv stack 
        self.unpool1 = nn.MaxUnpool2d((1, 2), stride=2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.unpool2 = nn.MaxUnpool2d((1, 2), stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.unpool3 = nn.MaxUnpool2d((1, 2), stride=2)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.unpool4 = nn.MaxUnpool2d((1, 2), stride=2)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                1024, self.n_features, kernel_size=(1, 5)
            ),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_domain: torch.Tensor,
        z_activity: torch.Tensor,
        pool_indices: List[torch.Tensor],
        pool_sizes: List[torch.Size],
        z_shared: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_domain:    (B, latent_dim_domain)
            z_activity:  (B, latent_dim_activity)
            z_shared:    (B, latent_dim_shared) or None
            pool_indices: list of pooling indices (len=4)
            pool_sizes:   list of tensor sizes before pooling (len=4)

        Returns:
            x_recon: (B, 1, T, F) or equivalent
        """

        #  Concatenate latents 
        """        if self.latent_dim_shared > 0:
            if z_shared is None:
                raise ValueError("z_shared must be provided when latent_dim_shared > 0")
            z = torch.cat([z_domain, z_shared, z_activity], dim=1)
        else:"""
        z = torch.cat([z_domain, z_activity], dim=1)

        #  Project to conv feature map 
        h = self.fc(z)
        h = h.view(-1, 64, 1, 2)

        #  Unpool + deconv 
        h = self.unpool1(h, pool_indices[3], output_size=pool_sizes[2])
        h = self.deconv1(h)

        h = self.unpool2(h, pool_indices[2], output_size=pool_sizes[1])
        h = self.deconv2(h)

        h = self.unpool3(h, pool_indices[1], output_size=pool_sizes[0])
        h = self.deconv3(h)

        h = self.unpool4(h, pool_indices[0])
        h = self.deconv4(h)

        # (B, C, 1, T) → (B, 1, T, C)
        return h.permute(0, 2, 3, 1)




class QD(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()
        self.fc = nn.Linear(zd_dim, d_dim)

    def forward(self, z):
        return self.fc(z)
    
class QY(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super().__init__()
        self.fc = nn.Linear(zy_dim, y_dim)

    def forward(self, z):
        return self.fc(z)

class GILE(nn.Module):
    def __init__(
        self,
        input_dim,
        activity_classes,
        domain_classes,
        latent_dim=32,
        beta_domain=1.0,
        beta_activity=1.0,
    ):
        super().__init__()
        self.aux_loss_multiplier_d =1000
        self.aux_loss_multiplier_y =1000
        self.weight_true = 1000
        self.weight_false = 1000
        self.beta_d = 0.002
        self.beta_y = 10

        self.zd_dim = 50 #d_AE
        self.zx_dim = 0
        self.zy_dim = 50 #d_AE
        self.d_dim = 4 #n_domains
        self.x_dim = 0
        self.y_dim = 18 #class
        # Encoders
        self.activity_encoder = QZY(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) #qzy
        self.domain_encoder = QZD(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) # qzd

        # Priors
        self.activity_prior = ClassPrior(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) #pzy
        self.domain_prior = DomainPrior(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) #pzd

        # Decoder
        self.decoder = PX(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) #

        # Disentangling classifiers
        self.activity_classifier = QY(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) # qy
        self.domain_classifier = QD(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim) # qd

        # KL weights
        self.beta_activity = beta_activity
        self.beta_domain = beta_domain

    def forward(self, x, y, d):
        x = torch.unsqueeze(x, 1)
        d = d.long().view(-1)   # or d = d.squeeze(-1)
        y = y.long().view(-1)   # or y = y.squeeze(-1)

        #encode
        mu_y, sigma_y, pool_idx_y, pool_sizes_y = self.activity_encoder(x)
        mu_d, sigma_d, _, _ = self.domain_encoder(x)

        # reparameterization
        qzd = dist.Normal(mu_d,sigma_d)
        qzy = dist.Normal(mu_y,sigma_y)

        zy_q = qzy.rsample()
        zd_q = qzd.rsample()

        # decode

        x_recon = self.decoder(
            z_domain=zd_q,
            z_activity=zy_q,
            pool_indices=pool_idx_y,   
            pool_sizes=pool_sizes_y,
        )
        #priors
        zd_p_loc, zd_p_scale = self.domain_prior(d)
        zy_p_loc, zy_p_scale = self.activity_prior(y)

        #reparameterization
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        y_hat  = self.activity_classifier(zy_q)
        d_hat  = self.domain_classifier(zd_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, _,_,_, qzy, pzy, zy_q

    
  
    def compute_loss( self, x, y, d ):
        
        d = d.long().view(-1)
        y = y.long().view(-1)
        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(x,y,d)
        x = torch.unsqueeze(x, 1)

        CE_x = F.mse_loss(x_recon, x.float())
        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        
        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))
        CE_d = F.cross_entropy(d_hat, d, reduction='sum')
        CE_y = F.cross_entropy(y_hat, y, reduction='sum')

        return CE_x \
               - self.beta_d * zd_p_minus_zd_q \
               - self.beta_y * zy_p_minus_zy_q \
               + self.aux_loss_multiplier_d * CE_d \
               + self.aux_loss_multiplier_y * CE_y,\
               CE_y



    def compute_loss_false(
        self,
        x,
        y,
        d,
        
    ):
        """
        IE / false step (optimizer_ie):
        train ONLY the IE heads to be good at predicting the wrong thing.
        -> detach latents so encoders dont move in this step
        """
        d = d.long()
        y = y.long()

        pred_d, pred_y, pred_d_false, pred_y_false = self.classifier(x)

        loss_classify_true = self.weight_true * (F.cross_entropy(pred_d, d, reduction='sum') + F.cross_entropy(pred_y, y, reduction='sum'))
        loss_classify_false = self.weight_false * (F.cross_entropy(pred_d_false, d, reduction='sum') + F.cross_entropy(pred_y_false, y, reduction='sum'))

        loss = loss_classify_true - loss_classify_false

        loss.requires_grad = True

        return loss

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():

            x = torch.unsqueeze(x, 1)

            zd_q_loc, zd_q_scale, _, _ = self.domain_encoder(x)
            zd = zd_q_loc
            alpha = F.softmax(self.domain_classifier(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha.size())
            d = d.scatter_(1, ind, 1.0)

            zy_q_loc, zy_q_scale, _, _ = self.activity_encoder.forward(x)
            zy = zy_q_loc
            alpha = F.softmax(self.activity_classifier(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha.size())
            y = y.scatter_(1, ind, 1.0)

            alpha_y2d = F.softmax(self.domain_classifier(zy), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_y2d, 1)
            # convert the digit(s) to one-hot tensor(s)
            d_false = x.new_zeros(alpha_y2d.size())
            d_false = d_false.scatter_(1, ind, 1.0)

            alpha_d2y = F.softmax(self.activity_classifier(zd), dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_d2y, 1)

            # convert the digit(s) to one-hot tensor(s)
            y_false = x.new_zeros(alpha_d2y.size())
            y_false = y_false.scatter_(1, ind, 1.0)

        return d, y, d_false, y_false
    
def kl_divergence(logvar_q, logvar_p, mu_q, mu_p):
    # returns mean KL over batch
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
    return kl.sum(dim=1).mean()