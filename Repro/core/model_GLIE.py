import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # stability
        logvar = torch.clamp(logvar, -10, 10)

        z = self.sample(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim_activity, latent_dim_domain, output_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim_activity + latent_dim_domain, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_activity, z_domain):
        z = torch.cat([z_activity, z_domain], dim=1)
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
    ):
        super().__init__()

        # Encoders
        self.activity_encoder = ProbabilisticEncoder(input_dim, latent_dim)
        self.domain_encoder = ProbabilisticEncoder(input_dim, latent_dim)

        # Decoder
        self.decoder = Decoder(
            latent_dim_activity=latent_dim,
            latent_dim_domain=latent_dim,
            output_dim=input_dim,
        )

        # Disentangling classifiers
        self.activity_classifier = LatentClassifier(latent_dim, activity_classes)
        self.domain_classifier = LatentClassifier(latent_dim, domain_classes)

    def forward(self, x):
        z_activity, mu_a, logvar_a = self.activity_encoder(x)
        z_domain, mu_d, logvar_d = self.domain_encoder(x)

        x_recon = self.decoder(z_activity, z_domain)

        activity_logits = self.activity_classifier(z_activity)
        domain_logits = self.domain_classifier(z_domain)

        return {
            "x_recon": x_recon,
            "z_activity": z_activity,
            "z_domain": z_domain,
            "mu_a": mu_a,
            "logvar_a": logvar_a,
            "mu_d": mu_d,
            "logvar_d": logvar_d,
            "activity_logits": activity_logits,
            "domain_logits": domain_logits,
        }

def kl_divergence(logvar_q, logvar_p, mu_q, mu_p):
    # returns mean KL over batch
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p)**2) / var_p - 1.0)
    return kl.sum(dim=1).mean()