from argparse import Namespace
import torch
from sklearn.metrics import f1_score

# from your model file (the spec-faithful one)
from ucihar_GILE import GILE  # this should contain the GILE(args) version

from oppor_dataloader_v2 import prep_domains_oppor
from ucihar_dataloader import prep_domains_ucihar
from shar_dataloader import prep_domains_shar_preprocessed


# -----------------------
# device
# -----------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# -----------------------
# dataloaders (example)
# IMPORTANT: loaders must yield x as (B, T, F) and y,d as label indices (B,)
# -----------------------
train_loaders, test_loader = prep_domains_ucihar()


# -----------------------
# build args for spec-faithful model
# NOTE: set these to your dataset properties
# -----------------------
args = Namespace(
    device=DEVICE,

    # dataset structure
    n_feature=9,     # F (UCihar)
    n_class=6,       # #activities
    n_domains=5,     # #domains

    # model hyperparams (author defaults)
    d_AE=50,         # latent dim for z_d and z_y
    x_dim=1152,      # kept for compatibility

    # loss weights
    aux_loss_multiplier_y=1000.0,
    aux_loss_multiplier_d=1000.0,
    beta_d=1.0,
    beta_x=0.0,
    beta_y=1.0,

    # IE/false-step weights (used by loss_function_false)
    weight_true=1000.0,
    weight_false=1000.0,
)


# -----------------------
# model
# -----------------------
model = GILE(args).to(DEVICE)


# -----------------------
# optimizers
# Author trains main + IE in separate steps; to match your pattern, use two optimizers.
# If you want strict author behavior, keep both on all params (as you already do).
# -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
false_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)


# -----------------------
# training
# -----------------------
def train_one_epoch(source_loaders):
    model.train()
    train_loss_sum = 0.0
    class_y_loss_sum = 0.0
    total = 0

    for source_loader in source_loaders:
        for x, y, d in source_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long().view(-1)
            d = d.to(DEVICE).long().view(-1)

            # ----- main step -----
            optimizer.zero_grad(set_to_none=True)
            loss_origin, ce_y = model.loss_function(d, x, y)
            loss_origin.backward()
            optimizer.step()

            # ----- IE/false step -----
            false_optimizer.zero_grad(set_to_none=True)
            loss_false = model.loss_function_false(args, d, x, y)
            loss_false.backward()
            false_optimizer.step()

            bs = y.size(0)
            total += bs

            # loss_origin and ce_y are tensors; accumulate as floats
            train_loss_sum += float(loss_origin.detach().cpu())
            class_y_loss_sum += float(ce_y.detach().cpu())

    # report per-sample averages (your old code did /total)
    avg_loss = train_loss_sum / max(total, 1)
    avg_ce_y = class_y_loss_sum / max(total, 1)
    return avg_loss, avg_ce_y


# -----------------------
# evaluation
# classifier returns one-hot predictions (author style)
# so we convert: pred = onehot.argmax(1)
# -----------------------
@torch.no_grad()
def get_accuracy(loaders, device, model, classifier_fn, batch_size):
    model.eval()

    pred_d_list, true_d_list = [], []
    pred_y_list, true_y_list = [], []
    pred_d_false_list, pred_y_false_list = [], []

    for loader in loaders:
        for xs, ys, ds in loader:
            xs = xs.to(device)
            ys = ys.to(device).long().view(-1)
            ds = ds.to(device).long().view(-1)

            d_onehot, y_onehot, d_false_onehot, y_false_onehot = classifier_fn(xs)

            # convert one-hot -> label indices
            d_pred = d_onehot.argmax(1)
            y_pred = y_onehot.argmax(1)
            d_false_pred = d_false_onehot.argmax(1)
            y_false_pred = y_false_onehot.argmax(1)

            pred_d_list.append(d_pred)
            pred_y_list.append(y_pred)
            pred_d_false_list.append(d_false_pred)
            pred_y_false_list.append(y_false_pred)

            true_d_list.append(ds)
            true_y_list.append(ys)

    d_true = torch.cat(true_d_list)
    y_true = torch.cat(true_y_list)
    d_pred = torch.cat(pred_d_list)
    y_pred = torch.cat(pred_y_list)
    d_false_pred = torch.cat(pred_d_false_list)
    y_false_pred = torch.cat(pred_y_false_list)

    # accuracies
    acc_d = (d_pred == d_true).float().mean().item()
    acc_y = (y_pred == y_true).float().mean().item()
    acc_d_false = (d_false_pred == d_true).float().mean().item()
    acc_y_false = (y_false_pred == y_true).float().mean().item()

    # macro F1
    d_macro_f1 = f1_score(d_true.cpu().numpy(), d_pred.cpu().numpy(), average="macro")
    y_macro_f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")

    return acc_d, acc_y, acc_d_false, acc_y_false, d_macro_f1, y_macro_f1


# -----------------------
# run training
# -----------------------
torch.manual_seed(10)

for epoch in range(150):
    avg_epoch_loss, avg_epoch_ce_y = train_one_epoch(train_loaders)

    train_acc_d, train_acc_y, train_acc_d_false, train_acc_y_false, _, _ = get_accuracy(
        train_loaders, DEVICE, model, model.classifier, batch_size=64
    )

    test_acc_d, test_acc_y, test_acc_d_false, test_acc_y_false, _, y_macro_f1 = get_accuracy(
        [test_loader], DEVICE, model, model.classifier, batch_size=64
    )

    print(
        f"Epoch {epoch:03d} | "
        f"Loss main: {avg_epoch_loss:8.4f} | "
        f"CE_y: {avg_epoch_ce_y:8.4f} | "
        f"Train ActAcc: {train_acc_y:5.3f} | "
        f"Train DomAcc: {train_acc_d:5.3f} | "
        f"Train FalseAct: {train_acc_y_false:5.3f} | "
        f"Train FalseDom: {train_acc_d_false:5.3f} | "
        f"Test ActAcc: {test_acc_y:5.3f} | "
        f"Test DomAcc: {test_acc_d:5.3f} | "
        f"Test F1 MacroAct: {y_macro_f1:5.3f}"
    )
