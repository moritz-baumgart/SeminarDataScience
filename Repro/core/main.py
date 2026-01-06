from torch.utils.data import DataLoader
from oppor_dataloader import build_opportunity_loader
from model_GLIE import kl_divergence, GILE

import torch
import torch.nn.functional as F


# hyperparameter
beta_kl = 0.01      # KL-Regularisierung
alpha_cls = 1.0    # Klassifikations-Loss
gamma_ie = 0.5     # Independent Excitation


# device
device = "cuda" if torch.cuda.is_available() else "cpu"


# dataloader (LOSO)
train_loader = build_opportunity_loader(
    domain_ids=["S1", "S2", "S3"],
    batch_size=64,
    shuffle=True,
    label_type="gestures",
)

test_loader = build_opportunity_loader(
    domain_ids=["S4"],
    batch_size=64,
    shuffle=False,
    label_type="gestures",
)


# model
model = GILE(
    input_dim=24*77,
    activity_classes=18,
    domain_classes=4,
    latent_dim=16,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        out = model(x)

        # reconstruction (ELBO)
        recon_loss = F.mse_loss(out["x_recon"], x)

        # priors p(z)=N(0,I)

        mu0_a = torch.zeros_like(out["mu_a"])
        lv0_a = torch.zeros_like(out["logvar_a"])
        mu0_d = torch.zeros_like(out["mu_d"])
        lv0_d = torch.zeros_like(out["logvar_d"])

        # KL(q || p)
        kl_a = kl_divergence(out["logvar_a"], lv0_a, out["mu_a"], mu0_a)
        kl_d = kl_divergence(out["logvar_d"], lv0_d, out["mu_d"], mu0_d)

        # negative ELBO (to minimize)
        elbo_loss = recon_loss + beta_kl * (kl_a + kl_d)

        # classification
        activity_loss = F.cross_entropy(out["activity_logits"], y)
        domain_loss   = F.cross_entropy(out["domain_logits"], d)

        classification_loss = activity_loss + domain_loss

        # independent excitation
        domain_from_activity = model.domain_classifier(out["z_activity"])
        activity_from_domain = model.activity_classifier(out["z_domain"])

        ie_loss = (
            F.cross_entropy(domain_from_activity, d)
            + F.cross_entropy(activity_from_domain, y)
        )

        # total loss
        loss = (
            elbo_loss
            + alpha_cls * classification_loss
            - gamma_ie * ie_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# training
for epoch in range(20):
    loss = train_one_epoch(model, train_loader)
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f}")
