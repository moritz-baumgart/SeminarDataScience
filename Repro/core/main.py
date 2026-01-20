from torch.utils.data import DataLoader
#from oppor_dataloader import build_opportunity_loader
from model_GLIE import kl_divergence, GILE
from utils import GradReverse
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
#from oppor_prepro_dataloader import build_opportunity_loader
from oppor_dataloader_v2 import build_opportunity_loader
# hyperparameter
aux_loss_multiplier_y = 100
aux_loss_multiplier_d = 100
beta_d = 0.002
beta_y = 10
weight_true = 100
weight_false = 10
latent_dim = 50

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_DOMAINS = ["S1", "S2", "S3"]
TARGET_DOMAIN  = "S4"

train_loaders = [
    build_opportunity_loader(
        domain_ids=[dom],      # IMPORTANT: list, not string
        batch_size=64,
        shuffle=True,
        expected_x_dim=30*77,
        label_type="gestures",
        balanced= True
    )
    for dom in SOURCE_DOMAINS
]

test_loader = build_opportunity_loader(
    domain_ids=[TARGET_DOMAIN],  # IMPORTANT: list, not string
    batch_size=4096,
    expected_x_dim=30*77,
    shuffle=False,
    label_type="gestures",
)


# model
model = GILE(
    input_dim=30*77,
    activity_classes=18,
    domain_classes=4,
    latent_dim=latent_dim,
    weight_true = weight_true,
    weight_false = weight_false,
    beta_domain= beta_d,
    beta_activity= beta_y,
    aux_loss_multiplier_y= aux_loss_multiplier_y,
    aux_loss_multiplier_d= aux_loss_multiplier_d,
    zx_dim= 0,

).to(device)
print(device)


main_params = (
    list(model.activity_encoder.parameters()) +
    list(model.domain_encoder.parameters()) +
    list(model.decoder.parameters()) +
    list(model.activity_prior.parameters()) +
    list(model.domain_prior.parameters())
)

ie_params = (
    #list(model.activity_encoder.parameters()) +
    #list(model.domain_encoder.parameters()) +
    list(model.activity_classifier.parameters()) +
    list(model.domain_classifier.parameters())

)# model.parameters()
opt_main = torch.optim.Adam(main_params, lr=1e-4, weight_decay=1e-3)
opt_ie   = torch.optim.Adam(ie_params, lr=1e-3, weight_decay=1e-3)

def train_one_epoch(model, train_loaders, opt_main, opt_ie):
    model.train()

    total_main = 0.0
    total_ie = 0.0
    total_dc = 0.0
    total = 0

    for loader in train_loaders:
        for x, y, d in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)

            batch_size = x.size(0)
            total += batch_size

            # --------------------------------------------------
            # (1) MAIN STEP
            # train encoder / decoder / priors / true classifiers
            # --------------------------------------------------
            opt_main.zero_grad(set_to_none=True)
            opt_ie.zero_grad(set_to_none=True)
            if not torch.isfinite(x).all():
                raise RuntimeError("Non-finite x coming from DataLoader")
            out = model(x, y, d)
            loss_main = model.loss_function_elbo(x=x, y=y, d=d, pred = out)
            loss_main.backward()
            opt_main.step()


            # after opt_main.step()
            assert_finite_module(model.domain_encoder, "domain_encoder")
            assert_finite_module(model.activity_encoder, "activity_encoder")

            # after opt_dc.step() / opt_ie.step()
            assert_finite_module(model.domain_encoder, "domain_encoder")
            
            total_main += loss_main.item() * batch_size

            # --------------------------------------------------
            # (2) IE / FALSE STEP
            # train ONLY IE heads (i.e., only params in opt_ie)
            # --------------------------------------------------
            

            # Important: recompute forward path if your loss_function_false
            # depends on stochastic sampling. If it's deterministic, you can keep it.
            if not torch.isfinite(x).all():
                raise RuntimeError("Non-finite x coming from DataLoader2")            
            out = model(x, y, d)
            loss_dc = model.loss_function_dc(x,y,d,out)
            loss_ie = model.loss_function_ie(x, y, d,out)


            loss_dcie= loss_dc+loss_ie


            loss_dcie.backward()
            opt_ie.step()


            # after opt_main.step()
            assert_finite_module(model.domain_encoder, "domain_encoder")
            assert_finite_module(model.activity_encoder, "activity_encoder")

            # after opt_dc.step() / opt_ie.step()
            assert_finite_module(model.domain_encoder, "domain_encoder")

            total_ie += loss_ie.item() * batch_size
            total_dc += loss_dc.item() * batch_size
    denom = max(total, 1)
    return {
        "loss_main": total_main / denom,
        "loss_ie": total_ie / denom,
        "loss_dc" : total_dc / denom,
        "loss_total": (total_main + total_ie) / denom,
    }
def assert_finite_module(m, name):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite param after step: {name}.{n}")
@torch.no_grad()
def evaluate(model, loaders):
    model.eval()
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    all_y_true, all_y_pred = [], []
    all_d_true, all_d_pred = [], []
    correct_y = correct_d = total = 0

    for loader in loaders:
        for x, y, d in loader:
            x, y, d = x.to(device), y.to(device), d.to(device)

            d_hat, y_hat, d_false, y_false = model.classify(x)

            y_pred = y_hat.argmax(dim=1)      # activity prediction
            d_pred = d_hat.argmax(dim=1)      # domain prediction (if you want domain accuracy)

           
            d_cross_pred = d_false.argmax(dim=1)  # domain from activity latent
            
            all_y_true.append(y.detach().cpu())
            all_y_pred.append(y_pred.detach().cpu())
            all_d_true.append(d.detach().cpu())
            all_d_pred.append(d_pred.detach().cpu())

            correct_y += (y_pred == y).sum().item()
            correct_d += (d_pred == d).sum().item()
            total += y.size(0)

    # concatenate for F1
    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()

    d_true = torch.cat(all_d_true).numpy()
    d_pred = torch.cat(all_d_pred).numpy()

    # F1 metrics 
    activity_f1_macro = f1_score(y_true, y_pred, average="macro")
    activity_f1_weighted = f1_score(y_true, y_pred, average="weighted")
    domain_f1_macro = f1_score(d_true, d_pred, average="macro")

    # accuracy metrics (authors' evaluation)
    #activity_accuracy = correct_y / max(total, 1)
    #domain_accuracy = correct_d / max(total, 1)

    # accuracy metrics (authors' evaluation: correct / (n_batches * batch_size))
    batch_size = getattr(loader, "batch_size", None) or 1
    n_batches = len(all_y_true)  # you append once per batch
    denom_authors = max(n_batches * batch_size, 1)

    activity_accuracy = correct_y / denom_authors
    domain_accuracy = correct_d / denom_authors
    return {
        "activity_f1_macro": activity_f1_macro,
        "activity_f1_weighted": activity_f1_weighted,
        "domain_f1_macro": domain_f1_macro,
        "activity_accuracy": activity_accuracy,
        "domain_accuracy": domain_accuracy,
    }

torch.manual_seed(10)
for epoch in range(500):
    loss = train_one_epoch(model, train_loaders, opt_main, opt_ie)
    metrics_train = evaluate(model, train_loaders)
    metrics_test  = evaluate(model, [test_loader])
    print(
            f"Epoch {epoch:02d} | "
            f"Train Loss main: {loss["loss_main"]:.4f} | "
            f"Train Loss ie: {loss["loss_ie"]:.4f} | "
            f"Train Loss dc: {loss["loss_dc"]:.4f} |" 
            f"Train Loss total: {loss["loss_total"]:.4f} | "
            f"Act F1 (macro): {metrics_test['activity_f1_macro']:.3f} | "
            f"Act F1 (weighted): {metrics_test['activity_f1_weighted']:.3f} | "
            f"activity_accuracy: {metrics_test['activity_accuracy']:.3f} | "

    )
"""        
    print(
            f"Act F1 (macro): {metrics_train['activity_f1_macro']:.3f} | "
            f"Act F1 (weighted): {metrics_train['activity_f1_weighted']:.3f} | "
            f"activity_accuracy: {metrics_train['activity_accuracy']:.3f} | "
            f"domain_accuracy: {metrics_train['domain_accuracy']:.3f} | "
            f"Domain F1 (weighted): {metrics_train['domain_f1_macro']:.3f}"

    )"""