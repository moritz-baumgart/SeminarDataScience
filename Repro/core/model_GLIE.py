from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


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


class ProbabilisticEncoder(nn.Module):
    """
    Generic VAE-style encoder:
        q(z | x) = N(mu(x), sigma(x)^2)

    Can be used for:
        - qzd (domain encoder)
        - qzy (class encoder)
        - qzx (shared encoder)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        layers = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
            ])
            dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (B, input_dim)
            sample: if False, returns mu instead of sampled z

        Returns:
            z:      (B, latent_dim)
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = self.backbone(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # numerical stability
        logvar = torch.clamp(logvar, -10.0, 10.0)

        if not sample:
            return mu, mu, logvar

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    p(x | z_d, z_y [, z_x])

    """

    def __init__(
        self,
        latent_dim_activity: int,
        latent_dim_domain: int,
        output_dim: int,
        latent_dim_shared: int = 0,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim_shared = latent_dim_shared
        total_latent_dim = latent_dim_activity + latent_dim_domain + latent_dim_shared

        self.net = nn.Sequential(
            nn.Linear(total_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z_activity: torch.Tensor,
        z_domain: torch.Tensor,
        z_shared: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_activity: (B, latent_dim_activity)
            z_domain:   (B, latent_dim_domain)
            z_shared:   (B, latent_dim_shared) or None

        Returns:
            x_recon: (B, output_dim)
        """
        if self.latent_dim_shared > 0:
            if z_shared is None:
                raise ValueError("z_shared must be provided when latent_dim_shared > 0")
            z = torch.cat([z_domain, z_shared, z_activity], dim=1)
        else:
            z = torch.cat([z_domain, z_activity], dim=1)

        return self.net(z)


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
        self.activity_encoder = ProbabilisticEncoder(input_dim, latent_dim)
        self.domain_encoder = ProbabilisticEncoder(input_dim, latent_dim)

        # Priors
        self.activity_prior = ClassPrior(activity_classes, latent_dim)
        self.domain_prior = DomainPrior(domain_classes, latent_dim)

        # Decoder
        self.decoder = Decoder(
            latent_dim_activity=latent_dim,
            latent_dim_domain=latent_dim,
            output_dim=input_dim,
        )

        # Disentangling classifiers
        self.activity_classifier = LatentClassifier(latent_dim, activity_classes)
        self.domain_classifier = LatentClassifier(latent_dim, domain_classes)
        
        # KL weights
        self.beta_activity = beta_activity
        self.beta_domain = beta_domain

    def forward(self, x, y, d):
        z_activity, mu_a, logvar_a = self.activity_encoder(x)
        z_domain, mu_d, logvar_d = self.domain_encoder(x)

        x_recon = self.decoder(z_activity, z_domain)

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
    
    def compute_loss(self, outputs, x, y, d):
        # Reconstruction
        recon_loss = F.mse_loss(outputs["x_recon"], x)

        # ----- PRIORS GO HERE -----
        mu_a_p, sigma_a_p = self.activity_prior(y)
        mu_d_p, sigma_d_p = self.domain_prior(d)

        logvar_a_p = torch.log(sigma_a_p ** 2)
        logvar_d_p = torch.log(sigma_d_p ** 2)

        kl_activity = kl_divergence(
            outputs["logvar_a"], logvar_a_p,
            outputs["mu_a"], mu_a_p,
        )

        kl_domain = kl_divergence(
            outputs["logvar_d"], logvar_d_p,
            outputs["mu_d"], mu_d_p,
        )

        # Auxiliary classification
        cls_activity = F.cross_entropy(outputs["activity_logits"], y)
        cls_domain = F.cross_entropy(outputs["domain_logits"], d)

        total_loss = (
            recon_loss
            + self.beta_activity * kl_activity
            + self.beta_domain * kl_domain
            + cls_activity
            + cls_domain
        )

        return total_loss

def kl_divergence(logvar_q, logvar_p, mu_q, mu_p):
    # returns mean KL over batch
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
    return kl.sum(dim=1).mean()