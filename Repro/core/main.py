from torch.utils.data import DataLoader
#from oppor_dataloader import build_opportunity_loader
from model_GLIE import kl_divergence, GILE
from utils import GradReverse
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from oppor_prepro_dataloader import build_opportunity_loader

# hyperparameter
beta_kl = 0.1    # KL-Regularisierung
alpha_cls = 1.0    # Klassifikations-Loss
gamma_ie = 0.2     # Independent Excitation
aux_y    = 1.0
aux_d    = 1.0

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cuda.is_built())

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
    input_dim=30*77,
    activity_classes=18,
    domain_classes=4,
    latent_dim=16,
).to(device)
print(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-3)


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        out = model(x, y, d)

        loss = model.compute_loss(x, y, d, out, beta_kl, aux_y, aux_d, gamma_ie)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_y_true = []
    all_y_pred = []

    all_d_true = []
    all_d_pred = []

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        out = model(x, y, d)

        # Activity predictions
        y_pred = torch.argmax(out["activity_logits"], dim=1)

        all_y_true.append(y.cpu())
        all_y_pred.append(y_pred.cpu())

        # (optional) Domain predictions
        d_pred = torch.argmax(out["domain_logits"], dim=1)
        all_d_true.append(d.cpu())
        all_d_pred.append(d_pred.cpu())

    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()

    d_true = torch.cat(all_d_true).numpy()
    d_pred = torch.cat(all_d_pred).numpy()

    activity_f1_macro = f1_score(y_true, y_pred, average="macro")
    activity_f1_weighted = f1_score(y_true, y_pred, average="weighted")

    domain_f1_macro = f1_score(d_true, d_pred, average="macro")

    return {
        "activity_f1_macro": activity_f1_macro,
        "activity_f1_weighted": activity_f1_weighted,
        "domain_f1_macro": domain_f1_macro,
    }
@torch.no_grad()
def eval(model, loader):
    model.eval()

    all_y_true = []
    all_y_pred = []

    all_d_true = []
    all_d_pred = []

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        out = model(x, y, d)

        # Activity predictions
        y_pred = torch.argmax(out["activity_logits"], dim=1)

        all_y_true.append(y.cpu())
        all_y_pred.append(y_pred.cpu())

        # (optional) Domain predictions
        d_pred = torch.argmax(out["domain_logits"], dim=1)
        all_d_true.append(d.cpu())
        all_d_pred.append(d_pred.cpu())

    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()

    d_true = torch.cat(all_d_true).numpy()
    d_pred = torch.cat(all_d_pred).numpy()

    activity_f1_macro = f1_score(y_true, y_pred, average="macro")
    activity_f1_weighted = f1_score(y_true, y_pred, average="weighted")

    domain_f1_macro = f1_score(d_true, d_pred, average="macro")

    return {
        "activity_f1_macro": activity_f1_macro,
        "activity_f1_weighted": activity_f1_weighted,
        "domain_f1_macro": domain_f1_macro,
    }


# training
for epoch in range(500):
    loss = train_one_epoch(model, train_loader)
    metrics = evaluate(model, test_loader)
    print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {loss:.4f} | "
            f"Act F1 (macro): {metrics['activity_f1_macro']:.3f} | "
            f"Act F1 (weighted): {metrics['activity_f1_weighted']:.3f} | "
            f"Domain F1 (weighted): {metrics['domain_f1_macro']:.3f}"
        )