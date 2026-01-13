from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple
from utils import GradReverse
from typing import Tuple, List, Optional

class DomainPrior(nn.Module):
    """
    p(z_d | d)
    Conditional Gaussian prior for domain latent variable z_d.
    """

    def __init__(self, num_domains: int, latent_dim: int):
        super().__init__()

        self.num_domains = num_domains
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Linear(num_domains, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
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

    def __init__(self, num_classes: int, latent_dim: int):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Linear(num_classes, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.scale_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
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

    def __init__(
        self,
        *,
        n_features: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.n_features = n_features

        # Convolutional backbone 
        self.conv1 = nn.Conv2d(n_features, 1024, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(512, 128, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        #  Latent heads 
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_sigma = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            x: Tensor [B, C, 1, T]

        Returns:
            mu:    [B, latent_dim]
            sigma: [B, latent_dim]
            pool_indices: indices for unpooling in decoder
        """
        x = x.float()

        # Accept [B, T*F], [B, T, F], or [B, 1, T, F]
        if x.dim() == 2:
            # [B, T*F] → [B, T, F]
            B = x.size(0)
            x = x.view(B, -1, self.n_features)

        if x.dim() == 3:
            # [B, T, F] → [B, F, 1, T]
            x = x.permute(0, 2, 1).unsqueeze(2)

        if x.dim() != 4:
            raise RuntimeError(f"Unexpected input shape to QZD: {x.shape}")
        x = self.relu(self.conv1(x))
        x, idx1 = self.pool1(x)

        x = self.relu(self.conv2(x))
        x, idx2 = self.pool2(x)

        x = self.relu(self.conv3(x))
        x, idx3 = self.pool3(x)

        x = self.relu(self.conv4(x))
        x, idx4 = self.pool4(x)

        # flatten (batch, channels, 1, time) → (batch, features)
        x = x.flatten(start_dim=1)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        pool_indices = (idx1, idx2, idx3, idx4)

        return mu, sigma, pool_indices, None
    
class QZY(nn.Module):
    """
    q(z_y | x)
    Activity-specific latent encoder
    """

    def __init__(
        self,
        *,
        n_features: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.n_features = n_features

        # Convolutional backbone 
        self.conv1 = nn.Conv2d(n_features, 1024, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(512, 128, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool2 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool3 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)
        self.pool4 = nn.MaxPool2d((1, 2), stride=2, return_indices=True, ceil_mode=True)

        # Latent heads 
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_sigma = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Size],
    ]:
        """
        Args:
            x: Tensor of shape [B, 1, T, F] or equivalent

        Returns:
            mu:            [B, latent_dim]
            sigma:         [B, latent_dim]
            pool_indices:  list of pooling indices
            pool_sizes:    list of tensor sizes before pooling
        """

        x = x.float()

        if x.dim() == 2:
            # [B, T*F] -> [B, T, F]
            B = x.size(0)
            x = x.view(B, -1, self.n_features)

        if x.dim() == 3:
            # [B, T, F] -> [B, F, 1, T]
            x = x.permute(0, 2, 1).unsqueeze(2)

        if x.dim() != 4:
            raise RuntimeError(f"Unexpected input shape to QZY: {x.shape}")

        x = self.relu(self.conv1(x))
        x, idx1 = self.pool1(x)
        size1 = x.size()

        x = self.relu(self.conv2(x))
        x, idx2 = self.pool2(x)
        size2 = x.size()

        x = self.relu(self.conv3(x))
        x, idx3 = self.pool3(x)
        size3 = x.size()

        x = self.relu(self.conv4(x))
        x, idx4 = self.pool4(x)
        size4 = x.size()

        # flatten for latent heads
        x = x.flatten(start_dim=1)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x) + 1e-7

        pool_indices = [idx1, idx2, idx3, idx4]
        pool_sizes = [size1, size2, size3, size4]

        return mu, sigma, pool_indices, pool_sizes
    

class PX(nn.Module):
    """
    p(x | z_d, z_y [, z_x])

    Decoder using unpooling + transposed convolutions.
    """

    def __init__(
        self,
        *,
        n_features: int,
        latent_dim_domain: int,
        latent_dim_activity: int,
        latent_dim_shared: int = 0,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.latent_dim_shared = latent_dim_shared

        total_latent_dim = latent_dim_domain + latent_dim_activity + latent_dim_shared

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
                1024, n_features, kernel_size=(1, 5)
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
        if self.latent_dim_shared > 0:
            if z_shared is None:
                raise ValueError("z_shared must be provided when latent_dim_shared > 0")
            z = torch.cat([z_domain, z_shared, z_activity], dim=1)
        else:
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




class LatentClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

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


        # Encoders
        self.activity_encoder = QZY(n_features=77, latent_dim=latent_dim, )
        self.domain_encoder = QZD(n_features=77, latent_dim=latent_dim,)

        # Priors
        self.activity_prior = ClassPrior(activity_classes, latent_dim)
        self.domain_prior = DomainPrior(domain_classes, latent_dim)

        # Decoder
        self.decoder = PX(n_features=77, latent_dim_domain=latent_dim, latent_dim_activity=latent_dim,)

        # Disentangling classifiers
        self.activity_classifier = LatentClassifier(latent_dim, activity_classes)
        self.domain_classifier = LatentClassifier(latent_dim, domain_classes)

        self.activity_classifier_ie = LatentClassifier(latent_dim, activity_classes)  # DCy(z_d)
        self.domain_classifier_ie   = LatentClassifier(latent_dim, domain_classes)    # DCd(z)

        # KL weights
        self.beta_activity = beta_activity
        self.beta_domain = beta_domain

    def forward(self, x, y, d):
        mu_a, sigma_a, pool_idx_a, pool_sizes_a = self.activity_encoder(x)
        mu_d, sigma_d, _, _ = self.domain_encoder(x)

        # reparameterization
        eps_a = torch.randn_like(sigma_a)
        eps_d = torch.randn_like(sigma_d)

        z_activity = mu_a + eps_a * sigma_a
        z_domain   = mu_d + eps_d * sigma_d

        logvar_a = torch.log(sigma_a ** 2 + 1e-8)
        logvar_d = torch.log(sigma_d ** 2 + 1e-8)

        x_recon = self.decoder(
            z_domain=z_domain,
            z_activity=z_activity,
            pool_indices=pool_idx_a,   # or whichever path you reconstruct from
            pool_sizes=pool_sizes_a,
        )

        activity_logits = self.activity_classifier(z_activity)
        domain_logits = self.domain_classifier(z_domain)

        return {
            "x_recon": x_recon,
            "mu_a": mu_a,
            "logvar_a": logvar_a,
            "mu_d": mu_d,
            "logvar_d": logvar_d,
            "activity_logits": activity_logits,
            "domain_logits": domain_logits,
            "z_domain": z_domain,
            "z_activity" : z_activity,
            "y": y,
            "d": d,
        }
    
  
    def compute_loss_main(
    self,
    x,
    y,
    d,
    pred,
    beta_kl,
    aux_y: float = 1.0,
    aux_d: float = 1.0,
    gamma_ie: float = 0.2,
):
        B = x.size(0)
        x_recon  = pred["x_recon"].reshape(B, -1)
        x_target = x.reshape(B, -1)
        recon_loss = F.mse_loss(x_recon, x_target, reduction="sum") / B

        mu_a_p, sigma_a_p = self.activity_prior(y)
        mu_d_p, sigma_d_p = self.domain_prior(d)

        std_a_q = torch.exp(0.5 * pred["logvar_a"])
        std_d_q = torch.exp(0.5 * pred["logvar_d"])

        q_a = dist.Independent(dist.Normal(pred["mu_a"], std_a_q), 1)
        q_d = dist.Independent(dist.Normal(pred["mu_d"], std_d_q), 1)

        p_a = dist.Independent(dist.Normal(mu_a_p, sigma_a_p), 1)
        p_d = dist.Independent(dist.Normal(mu_d_p, sigma_d_p), 1)

        kl_a_neg = (p_a.log_prob(pred["z_activity"]) - q_a.log_prob(pred["z_activity"])).mean()
        kl_d_neg = (p_d.log_prob(pred["z_domain"])   - q_d.log_prob(pred["z_domain"])).mean()

        ce_y = F.cross_entropy(pred["activity_logits"], y, reduction="sum") / B
        ce_d = F.cross_entropy(pred["domain_logits"],   d, reduction="sum") / B
        dc_loss = aux_y * ce_y + aux_d * ce_d

        elbo_loss = recon_loss - beta_kl * (kl_a_neg + kl_d_neg)

        # IE (adversarial) -> GRL rein, vorzeichen bleibt "+"
        z_d_grl = GradReverse.apply(pred["z_domain"],   1.0)  # z_domain -> predict y
        z_y_grl = GradReverse.apply(pred["z_activity"], 1.0)  # z_activity -> predict d

        logits_y_from_zd = self.activity_classifier_ie(z_d_grl)
        logits_d_from_zy = self.domain_classifier_ie(z_y_grl)

        ie_y = F.cross_entropy(logits_y_from_zd, y, reduction="sum") / B
        ie_d = F.cross_entropy(logits_d_from_zy, d, reduction="sum") / B
        ie_loss = ie_y + ie_d

        # main objective: minimize elbo + dc, and (via GRL) remove leakage
        total_main = elbo_loss + dc_loss #+ gamma_ie * ie_loss
        return total_main



    def compute_loss_ie(
        self,
        y,
        d,
        pred,
    ):
        """
        IE / false step (optimizer_ie):
        train ONLY the IE heads to be good at predicting the wrong thing.
        -> detach latents so encoders dont move in this step
        """
        z_d = pred["z_domain"].detach()
        z_y = pred["z_activity"].detach()

        logits_y_from_zd = self.activity_classifier_ie(z_d)
        logits_d_from_zy = self.domain_classifier_ie(z_y)

        ie_y = F.cross_entropy(logits_y_from_zd, y, reduction="mean")
        ie_d = F.cross_entropy(logits_d_from_zy, d, reduction="mean")

        return ie_y + ie_d


def kl_divergence(logvar_q, logvar_p, mu_q, mu_p):
    # returns mean KL over batch
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
    return kl.sum(dim=1).mean()
