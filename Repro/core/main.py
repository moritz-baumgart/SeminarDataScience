from torch.utils.data import DataLoader
from oppor_dataloader import build_opportunity_loader
from model_GLIE import kl_divergence, GILE
from utils import GradReverse
import torch
import torch.nn.functional as F


# hyperparameter
beta_kl = 0.2      # KL-Regularisierung
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

        out = model(x, y, d)

        # reconstruction (ELBO)
        recon_loss = F.mse_loss(out["x_recon"], x)
        mu_a_p, sigma_a_p = model.activity_prior(y)
        mu_d_p, sigma_d_p = model.domain_prior(d)
        #print(mu_d_p.mean(), mu_a_p.mean())
        logvar_a_p = torch.log(sigma_a_p ** 2)
        logvar_d_p = torch.log(sigma_d_p ** 2)

        # KL(q || p)
        kl_a = kl_divergence(out["logvar_a"], logvar_a_p, out["mu_a"], mu_a_p)

        kl_d = kl_divergence(out["logvar_d"], logvar_d_p, out["mu_d"], mu_d_p)


        # negative ELBO (to minimize)
        elbo_loss = recon_loss + beta_kl * (kl_a + kl_d)

        # classification
        activity_loss = F.cross_entropy(out["activity_logits"], y)
        domain_loss   = F.cross_entropy(out["domain_logits"], d)

        classification_loss = activity_loss + domain_loss

        z_a_rev = GradReverse.apply(out["z_activity"], gamma_ie)
        z_d_rev = GradReverse.apply(out["z_domain"], gamma_ie)
        
        # independent excitation
        domain_from_activity = model.domain_classifier(z_a_rev)
        activity_from_domain = model.activity_classifier(z_d_rev)

        ie_loss = (
            F.cross_entropy(domain_from_activity, d)
            + F.cross_entropy(activity_from_domain, y)
        )

        loss = elbo_loss + alpha_cls * classification_loss + ie_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# training
for epoch in range(100):
    loss = train_one_epoch(model, train_loader)
    print(f"Epoch {epoch:02d} | Train Loss: {loss:.4f}")
