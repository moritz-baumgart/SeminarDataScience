from model_GLIE import GILE
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from oppor_dataloader_v2 import prep_domains_oppor
#from ucihar_dataloader import prep_domains_ucihar_preprocessed
from shar_dataloader import prep_domains_shar_preprocessed
# hyperparameter
beta_kl = 0.1    # KL-Regularisierung
alpha_cls = 1.0    # Klassifikations-Loss
gamma_ie = 0.2     # Independent Excitation
aux_y    = 1.0
aux_d    = 1.0

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# dataloader (LOSO)
SOURCE_DOMAINS = ["S1", "S2", "S3"]
TARGET_DOMAIN  = "S4"


train_loaders, test_loader = prep_domains_oppor()
# model
model = GILE(
    input_dim=77,
    activity_classes=18,
    domain_classes=4,
    latent_dim=16,
).to(device)
print(device)


#ie_params = (


main_params = []
for name, p in model.named_parameters():
    if name.startswith("activity_classifier_ie") or name.startswith("domain_classifier_ie"):
        continue
    main_params.append(p)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
false_optimizer   = torch.optim.Adam(model.parameters(),   lr=1e-3, weight_decay=1e-3)

def train_one_epoch(source_loaders):
    model.train()
    train_loss = 0.0
    total = 0.0
    epoch_class_y_loss = 0.0
    for source_loader in source_loaders:
        for x, y, d in source_loader:

            x = x.to(device)
            y = y.to(device)
            d = d.to(device)
            
            optimizer.zero_grad()
            false_optimizer.zero_grad()

            loss_origin, class_y_loss = model.compute_loss(x, y,d)

            loss_false = model.compute_loss_false(x, y,d)

            
            loss_origin.backward()
            optimizer.step()
            loss_false.backward()
            false_optimizer.step()

            train_loss += loss_origin
            epoch_class_y_loss += class_y_loss
            total += y.size(0)

    train_loss /= total
    epoch_class_y_loss /= total

    return train_loss, epoch_class_y_loss




def get_accuracy(source_loaders, DEVICE, model, classifier_fn, batch_size):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []
    predictions_d_false, predictions_y_false = [], []

    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:

                xs, ys, ds = xs.to(DEVICE), ys.to(DEVICE), ds.to(DEVICE)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y, pred_d_false, pred_y_false = classifier_fn(xs)
                predictions_d.append(pred_d)
                predictions_d_false.append(pred_d_false)
                actuals_d.append(ds)

                predictions_y.append(pred_y)
                predictions_y_false.append(pred_y_false)
                actuals_y.append(ys)

        # compute the number of accurate predictions
        accurate_preds_d = 0
        accurate_preds_d_false = 0
        for pred, act in zip(predictions_d, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d += v

        accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

        for pred, act in zip(predictions_d_false, actuals_d):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_d_false += v

        # calculate the accuracy between 0 and 1
        accuracy_d_false = (accurate_preds_d_false * 1.0) / (len(predictions_d_false) * batch_size)

        # compute the number of accurate predictions
        accurate_preds_y = 0
        accurate_preds_y_false = 0

        for pred, act in zip(predictions_y, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y += v

        accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)


        for pred, act in zip(predictions_y_false, actuals_y):
            _, num_pred = pred.max(1)
            v = torch.sum(num_pred == act)
            accurate_preds_y_false += v
            
        # calculate the accuracy between 0 and 1
        accuracy_y_false = (accurate_preds_y_false * 1.0) / (len(predictions_y_false) * batch_size)
          # domain macro F1
        d_true = torch.cat(actuals_d).cpu().numpy()
        d_pred = torch.cat([p.argmax(1) for p in predictions_d]).cpu().numpy()
        d_macro_f1 = f1_score(d_true, d_pred, average="macro")

        # activity macro F1
        y_true = torch.cat(actuals_y).cpu().numpy()
        y_pred = torch.cat([p.argmax(1) for p in predictions_y]).cpu().numpy()
        y_macro_f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy_d, accuracy_y, accuracy_d_false, accuracy_y_false, d_macro_f1, y_macro_f1



# training
torch.manual_seed(10)
for epoch in range(100):
    avg_epoch_loss, avg_epoch_class_y_loss = train_one_epoch(train_loaders)
    train_acc_d, train_acc_y, train_acc_d_false, train_acc_y_false, _,_ = get_accuracy(train_loaders, device, model, model.classifier, 64)
    test_acc_d, test_acc_y, test_acc_d_false, test_acc_y_false, _, y_macro_f1 = get_accuracy([test_loader], device, model, model.classifier, 64)
    print(
        f"Epoch {epoch:03d} | "
        f"Loss main: {avg_epoch_loss:8.3f} | "
        f"Loss ie: {avg_epoch_class_y_loss:8.3f} | "
        f"Train ActAcc: {train_acc_y:5.3f} | "
        f"Train DomAcc: {train_acc_d:5.3f} | "
        f"Train FalseAct: {train_acc_y_false:5.3f} | "
        f"Train FalseDom: {train_acc_d_false:5.3f} | "
        f"Test ActAcc: {test_acc_y:5.3f} | "
        f"Test DomAcc: {test_acc_d:5.3f} | "
        f"Test F1 MacroAct: {y_macro_f1:5.3f}"
    )