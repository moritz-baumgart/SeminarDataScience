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


ie_params = (
    list(model.activity_classifier_ie.parameters()) +
    list(model.domain_classifier_ie.parameters())
)

main_params = []
for name, p in model.named_parameters():
    if name.startswith("activity_classifier_ie") or name.startswith("domain_classifier_ie"):
        continue
    main_params.append(p)

opt_main = torch.optim.Adam(main_params, lr=1e-3, weight_decay=1e-3)
opt_ie   = torch.optim.Adam(ie_params,   lr=1e-3, weight_decay=1e-3)

def train_one_epoch(model, loader, opt_main, opt_ie):
    model.train()
    total_main = 0.0
    total_ie = 0.0

    for x, y, d in loader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)

        # (1) IE / false step
        # train ONLY IE heads, encoders must NOT move (detach is inside compute_loss_ie)
        out = model(x, y, d)

        loss_ie = model.compute_loss_ie(y, d, out)

        opt_ie.zero_grad(set_to_none=True)
        loss_ie.backward()
        opt_ie.step()

        total_ie += loss_ie.item()

        # (2) main step
        # train encoder/decoder/priors + true heads, and confuse IE heads via -gamma * ie_loss
        out = model(x, y, d)  # recompute forward (safer after step 1)

        loss_main = model.compute_loss_main(x, y, d, out, beta_kl, aux_y, aux_d, gamma_ie)

        opt_main.zero_grad(set_to_none=True)
        loss_main.backward()
        opt_main.step()

        total_main += loss_main.item()

    return {
        "loss_main": total_main / len(loader),
        "loss_ie": total_ie / len(loader),
        "loss_total": (total_main + total_ie) / len(loader),
    }



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
    loss = train_one_epoch(model, train_loader, opt_main, opt_ie)
    metrics = evaluate(model, test_loader)
    print(
            f"Epoch {epoch:02d} | "
            f"Train Loss main: {loss["loss_main"]:.4f} | "
            f"Train Loss ie: {loss["loss_ie"]:.4f} | "
            f"Train Loss total: {loss["loss_total"]:.4f} | "
            f"Act F1 (macro): {metrics['activity_f1_macro']:.3f} | "
            f"Act F1 (weighted): {metrics['activity_f1_weighted']:.3f} | "
            f"Domain F1 (weighted): {metrics['domain_f1_macro']:.3f}"
        )