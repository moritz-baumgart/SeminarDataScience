# shar_main.py (NO argparse) — works with the self-config GILE model
#
# expects:
#   - from model_GLIE import GILE   (the self-config version you asked for)
#   - from shar_dataloader import prep_domains_shar_preprocessed
# loader yields:
#   x: (B, T, F)  (e.g., (B,151,3))
#   y: (B,) int labels
#   d: (B,) int labels

import torch
from sklearn.metrics import f1_score

from shar_GILE import GILE
from shar_dataloader import prep_domains_shar_preprocessed


# -----------------------
# hyperparams (keep simple like your older scripts)
# -----------------------
SEED = 10
BATCH_SIZE = 64
N_EPOCH = 100
LR = 1e-3
WEIGHT_DECAY = 1e-3

# SHAR specs
N_FEATURE = 3
N_CLASS = 17
N_DOMAINS = 6
D_AE = 50
LEN_SW = 151  # not used by model, but useful for sanity checks


# -----------------------
# device
# -----------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def train_one_epoch(model, source_loaders, optimizer, false_optimizer):
    model.train()
    loss_sum = 0.0
    ce_y_sum = 0.0
    total = 0

    for loader in source_loaders:
        for x, y, d in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).long().view(-1)
            d = d.to(DEVICE).long().view(-1)

            # ---- main step ----
            optimizer.zero_grad(set_to_none=True)
            loss_main, ce_y = model.loss_function(d, x, y)
            loss_main.backward()
            optimizer.step()

            # ---- false / IE step ----
            false_optimizer.zero_grad(set_to_none=True)
            loss_false = model.loss_function_false(d, x, y)
            loss_false.backward()
            false_optimizer.step()

            bs = y.size(0)
            total += bs
            loss_sum += float(loss_main.detach().cpu())
            ce_y_sum += float(ce_y.detach().cpu())

    return loss_sum / max(total, 1), ce_y_sum / max(total, 1)


@torch.no_grad()
def get_accuracy(loaders, model, classifier_fn):
    model.eval()

    d_true_all, d_pred_all = [], []
    y_true_all, y_pred_all = [], []
    d_false_all, y_false_all = [], []

    for loader in loaders:
        for xs, ys, ds in loader:
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE).long().view(-1)
            ds = ds.to(DEVICE).long().view(-1)

            # classifier returns one-hot (author-style)
            d_oh, y_oh, d_false_oh, y_false_oh = classifier_fn(xs)

            d_pred = d_oh.argmax(1)
            y_pred = y_oh.argmax(1)
            d_false_pred = d_false_oh.argmax(1)
            y_false_pred = y_false_oh.argmax(1)

            d_true_all.append(ds)
            y_true_all.append(ys)

            d_pred_all.append(d_pred)
            y_pred_all.append(y_pred)

            d_false_all.append(d_false_pred)
            y_false_all.append(y_false_pred)

    d_true = torch.cat(d_true_all)
    y_true = torch.cat(y_true_all)
    d_pred = torch.cat(d_pred_all)
    y_pred = torch.cat(y_pred_all)
    d_false_pred = torch.cat(d_false_all)
    y_false_pred = torch.cat(y_false_all)

    acc_d = (d_pred == d_true).float().mean().item()
    acc_y = (y_pred == y_true).float().mean().item()
    acc_d_false = (d_false_pred == d_true).float().mean().item()
    acc_y_false = (y_false_pred == y_true).float().mean().item()

    d_macro_f1 = f1_score(d_true.cpu().numpy(), d_pred.cpu().numpy(), average="macro")
    y_weighted_f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="weighted")

    return acc_d, acc_y, acc_d_false, acc_y_false, d_macro_f1, y_weighted_f1


def main():
    torch.manual_seed(SEED)

    # dataloaders
    # NOTE: if your prep_domains_shar_preprocessed() does not accept batch_size,
    # remove the arg and ensure it internally uses BATCH_SIZE.
    train_loaders, test_loader = prep_domains_shar_preprocessed(batch_size=BATCH_SIZE)

    # optional sanity check (one batch)
    x0, y0, d0 = next(iter(train_loaders[0]))
    print("Sanity batch shapes:", x0.shape, y0.shape, d0.shape)
    # expected: (B, 151, 3) (B,) (B,)

    # model (self-config, no args namespace)
    model = GILE(
        n_feature=N_FEATURE,
        n_class=N_CLASS,
        n_domains=N_DOMAINS,
        d_AE=D_AE,
        beta_d=1.0,
        beta_x=1.0,
        beta_y=1.0,
        aux_loss_multiplier_y=1000.0,
        aux_loss_multiplier_d=1000.0,
        weight_true=1000.0,
        weight_false=1000.0,
        device=DEVICE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    false_optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(N_EPOCH):
        avg_epoch_loss, avg_epoch_ce_y = train_one_epoch(model, train_loaders, optimizer, false_optimizer)

        train_acc_d, train_acc_y, train_acc_d_false, train_acc_y_false, _, _ = get_accuracy(
            train_loaders, model, model.classifier
        )
        test_acc_d, test_acc_y, test_acc_d_false, test_acc_y_false, _, y_weighted_f1 = get_accuracy(
            [test_loader], model, model.classifier
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
            f"Test F1 WeightedAct: {y_weighted_f1:5.3f}"
        )


if __name__ == "__main__":
    main()
