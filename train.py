import warnings
import torchvision

warnings.filterwarnings("ignore")
torchvision.disable_beta_transforms_warning()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import time
import os
import random
from torchvision import datasets, models
from datetime import datetime
from torcheval.metrics.functional import binary_f1_score

from utils import (
    test_binary_model,
    validation_transform,
    inv_normalize,
    binary_confusion_matrix,
    matthews_correlation_coefficient,
    stats_to_tensorboard,
    adapt_network_classes,
    CosineAnnealingWarmRestarts,
)

from torch.backends import cudnn

cudnn.benchmark = (
    False  # Setting to True breaks determinism of random generators despite fixed seed
)
torch.backends.cudnn.deterministic = True

network_name_to_model = {
    "EfficientNet-v2-s": models.efficientnet_v2_s,
    "MobileNet-v3-small": models.mobilenet_v3_small,
    "MobileNet-v2": models.mobilenet_v2,
    "EfficientNet-b0": models.efficientnet_b0,
    "MobileNet-v3-large": models.mobilenet_v3_large,
    "EfficientNet-b1": models.efficientnet_b1,
    "ResNet50": models.resnet50,
    "Densenet": models.densenet161,
    "EfficientNet-v2-m": models.efficientnet_v2_m,
    "VisionTransformer-B": models.vit_b_16,
    "EfficientNet-v2-l": models.efficientnet_v2_l,
    "VGG16": models.vgg16,
    "VisionTransformer-L": models.vit_l_16,
}

TEST_EVERY_EPOCH = False
LOGGING = False  # Requires Tensorboard
DTYPE = torch.bfloat16

if LOGGING:
    from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_training(
    exp_name,
    dataset_path,
    network_name,
    hyperparameters,
    train_transform,
    seed,
    statistics_filename,
    test_datasets=None,
    save_models=True,
):
    identifier = exp_name + " " + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    cohort = os.path.split(dataset_path)[1]
    p = hyperparameters
    training_stats_summary = {
        "network": network_name,
        "dataset": cohort,
        "augmentation strategy": p["augmentation"],
        "hyperparameter set": p["hyperparameter set"],
    }

    if (
        network_name
        in [
            "EfficientNet-v2-l",
            "VGG16",
            "VisionTransformer-L",
        ]
        and p["batch size"] >= 8
    ):
        train_batch_size = 2
        p["gradient accumulation"] = int(p["batch size"] / 2)
    else:
        train_batch_size = p["batch size"]

    for key, value in training_stats_summary.items():
        print(f"{key}: {value}")

    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    res = p["resolution"]

    image_datasets = {
        "train": datasets.ImageFolder(
            os.path.join(dataset_path, "train"), train_transform
        ),
        "validation": datasets.ImageFolder(
            os.path.join(dataset_path, "validation"), validation_transform(res)
        ),
        "test": datasets.ImageFolder(
            os.path.join(dataset_path, "test"), validation_transform(res)
        ),
    }

    dl_allocation = {"train": 8, "validation": 2, "test": 1}
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=train_batch_size,
            shuffle=phase == "train",
            num_workers=dl_allocation[phase],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
            worker_init_fn=seed_worker,
            generator=g,
        )
        for phase in ["train", "validation", "test"]
    }

    if test_datasets:
        external_test_datasets = {}
        for test_ds in test_datasets:
            external_test_datasets[test_ds] = torch.utils.data.ConcatDataset(
                [
                    datasets.ImageFolder(
                        os.path.join(test_ds, "train"),
                        validation_transform(res),
                    ),
                    datasets.ImageFolder(
                        os.path.join(test_ds, "validation"),
                        validation_transform(res),
                    ),
                    datasets.ImageFolder(
                        os.path.join(test_ds, "test"), validation_transform(res)
                    ),
                ]
            )

        external_test_dataloaders = {
            ds: torch.utils.data.DataLoader(
                external_test_datasets[ds],
                batch_size=p["batch size"],
                shuffle=False,
                num_workers=2,
                prefetch_factor=1,
            )
            for ds in external_test_datasets
        }

    num_per_class = torch.unique(
        torch.tensor(image_datasets["train"].targets), return_counts=True
    )[1]

    print(
        "Train images per class in dataset " + cohort,
        str([int(n) for n in num_per_class]),
    )

    print("Classes: normal/abnormal")
    print("Class to index:", image_datasets["test"].class_to_idx)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")

    def train_model(
        model,
        criterion,
        optimizer,
        data,
        scheduler=None,
        num_epochs=1,
        writer=None,
    ):
        since = time.time()
        best_model_copy = model.state_dict()

        best_val_loss = 1e8
        best_mcc = 0
        best_acc = 0
        best_epoch = 0
        epoch_range = range(num_epochs + 1)
        early_stop = False

        for epoch in epoch_range:
            if early_stop:
                break

            epoch_t0 = time.time()
            print(f"Epoch {epoch}/{p['max epochs']}")
            print("-" * 12)

            phases = (
                ["train", "validation", "test"]
                if TEST_EVERY_EPOCH
                else ["train", "validation"]
            )

            for phase in phases:
                if epoch == 0 and phase == "train":
                    continue  # Do a validation phase on untrained model
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = []
                running_corrects = 0
                running_n_samples = 0
                pred_batches = []
                label_batches = []

                # Iterate over one epoch of dataset.
                for it, batch in enumerate(data[phase]):
                    inputs, labels = batch

                    # If we are augmenting each image multiple times in the same batch
                    if len(inputs.shape) > 4:
                        labels = labels.repeat_interleave(inputs.shape[1])
                        inputs = inputs.view(-1, 3, res, res)

                    inputs = inputs.to("cuda")
                    if p["label smoothing"]:
                        smoothed = (labels * (1 - (2 * p["label smoothing"]))) + p[
                            "label smoothing"
                        ]
                        labels_cuda = smoothed.float().view(-1, 1).to("cuda")
                    else:
                        labels_cuda = labels.float().view(-1, 1).to("cuda")

                    # backward + optimize only if in training phase
                    if phase == "train":
                        with torch.autocast(device_type="cuda", dtype=DTYPE):
                            logits = model(inputs)
                            loss = criterion(logits, labels_cuda)

                        if p["gradient accumulation"]:
                            loss = loss / p["gradient accumulation"]

                        loss = loss.mean()
                        loss.backward()
                        # print("Loss backward", it)
                        if (
                            not p["gradient accumulation"]
                            or ((it + 1) % p["gradient accumulation"]) == 0
                            or (it + 1) == len(data[phase])
                        ):
                            # if ((it + 1) % p['gradient accumulation']) == 0:
                            # print("Optimizer step")
                            optimizer.step()
                            optimizer.zero_grad()

                        scheduler.step(epoch + (it / len(data["train"])))

                    else:
                        with torch.autocast(device_type="cuda", dtype=DTYPE):
                            logits = model(inputs)
                            loss = criterion(logits, labels_cuda).mean()

                        pred_batches.append(F.sigmoid(logits.detach()).flatten())
                        label_batches.append(labels_cuda.flatten())

                    # Statistics
                    batch_preds = F.sigmoid(logits.detach()).flatten()
                    batch_truth = labels.to(device).flatten().long()
                    running_loss.append(loss.item())
                    running_corrects += (
                        batch_preds.round().long() == batch_truth
                    ).sum()
                    running_n_samples += len(batch_preds)

                # Phase finished, logging statistics
                epoch_loss = sum(running_loss) / len(running_loss)
                epoch_acc = running_corrects / running_n_samples

                if phase == "train":
                    if LOGGING:
                        try:
                            writer.add_scalar(
                                "training loss",
                                epoch_loss,
                                epoch,
                            )
                            writer.add_scalar(
                                "training accuracy",
                                epoch_acc,
                                epoch,
                            )
                        except Exception as error:
                            print(
                                "An exception occurred:",
                                type(error).__name__,
                                "-",
                                error,
                            )
                else:
                    x = torch.cat(pred_batches)
                    y = torch.round(torch.cat(label_batches))
                    epoch_mcc = matthews_correlation_coefficient(x, y)
                    epoch_F1 = binary_f1_score(x, y).item()
                    TP, FP, TN, FN = binary_confusion_matrix(x, y)
                    sensitivity = TP / (TP + FN)
                    specificity = TN / (TN + FP)

                    if LOGGING:
                        stats_to_tensorboard(
                            writer,
                            phase,
                            epoch,
                            epoch_loss,
                            epoch_acc,
                            epoch_F1,
                            epoch_mcc,
                            sensitivity,
                            specificity,
                            optimizer.param_groups[-1]["lr"],
                        )

                print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")
                if phase == "validation":
                    # Since MCC is often equal for several epochs when evaluating on small datasets,
                    # there is some ambiguity on how to select the best model
                    if p["best model metric"] == "MCC":
                        new_best_epoch = epoch_mcc > best_mcc  # or (
                        #    epoch_mcc == best_mcc and epoch_loss < best_val_loss
                        # )
                        # new_best_epoch = epoch_mcc >= best_mcc

                    elif p["best model metric"] == "BCE":
                        new_best_epoch = epoch_loss < best_val_loss
                        # elif p["best model metric"] == "AUC":
                        #    auc = binary_auroc(input=x, target=y).item()
                        #    print(auc)
                    else:
                        new_best_epoch = False

                    if new_best_epoch:
                        best_epoch = epoch
                        # Keep a copy of the best model to use for evaluation at the end of training
                        best_model_copy = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                        # best_model_copy = copy.deepcopy(best_model_copy)

                    # Trigger for early stop
                    elif (
                        epoch >= best_epoch + p["patience"] - 1
                        and epoch >= p["min epochs"] - 1
                    ):
                        early_stop = True

                    best_acc = max(best_acc, epoch_acc)
                    best_mcc = max(best_mcc, epoch_mcc)
                    best_val_loss = min(best_val_loss, epoch_loss)

                    print(
                        "Epoch time:",
                        int(time.time() - epoch_t0),
                        "LR:",
                        "{:.2e}".format(optimizer.param_groups[-1]["lr"]),
                        "val F1:",
                        round(epoch_F1, 2),
                        "val mcc:",
                        round(epoch_mcc, 2),
                    )
            print()

        time_elapsed = time.time() - since
        message = "Training complete"
        print(message + f" in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best mcc : {best_mcc:.3f}\n\n")
        print(f"Epoch of best model: {best_epoch}\n\n")

        # load best model weights
        model.load_state_dict(best_model_copy)
        return model, round(time_elapsed), epoch, best_epoch

    training_stats_summary["test metrics"] = {}
    training_stats_summary["validation metrics"] = {}
    stat_path = os.path.join("statistics", statistics_filename)
    if os.path.exists(stat_path):
        with open(stat_path) as f:
            all_stats = json.load(f)
    else:
        all_stats = {}

    for repeat in range(p["repeats"]):
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print("\n")
        if LOGGING:
            writer = SummaryWriter(
                f"runs/{exp_name}_{network_name}_{cohort}_{date_time}"
            )

            # Make image grids for tensorboard
            for phase in ["train", "validation"]:
                dataiter = iter(dataloaders[phase])
                images, _ = next(dataiter)
                if len(images.shape) == 5:
                    images = images.view(-1, 3, res, res)
                inp = inv_normalize(images[:24]).float()
                img_grid = torchvision.utils.make_grid(inp, nrow=6)
                writer.add_image(phase + " batch", img_grid)
        else:
            writer = None

        torch.cuda.empty_cache()
        # Initializing from imagenet pretrained model
        if network_name in network_name_to_model:
            model = network_name_to_model[network_name](weights="DEFAULT")
        else:
            raise Exception("Unsupported network")
            # model = timm.create_model(network_name, pretrained=True)
        adapt_network_classes(network_name, model, 1)

        if p["dropout"] != "default":
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p["dropout"]

        print(
            f"\nStarting training {network_name} on cohort {cohort}, repeat {repeat+1} of {p['repeats']}"
        )

        # summary(model, input_size=(BATCH_SIZE, 3, 224, 224))
        model = model.to(device)

        normal_count, abnormal_count = torch.unique(
            torch.tensor(dataloaders["train"].dataset.targets), return_counts=True
        )[1]
        print(f"Normal count: {normal_count}, abnormal count: {abnormal_count}")

        positive_weight = (
            normal_count / abnormal_count if p["class weighting"] else None
        )
        if positive_weight:
            print("Weight:", round(positive_weight.item(), 3))

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight,
            reduction="none",
        )

        def create_optimizer_and_scheduler(opt_type):
            if opt_type == "ADAMW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=p["learning rate"],
                    weight_decay=p["weight decay"],
                    amsgrad=p["AMSGrad (ADAMW)"],
                    fused=True,
                )
            else:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=p["learning rate"],
                    momentum=p["momentum (SGD)"],
                    weight_decay=p["weight decay"],
                    nesterov=p["Nesterov (SGD)"],
                )

            if p["LR schedule"] == "cosine":
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, p["LR restart interval"], eta_min=1e-8, T_mult=1.5
                )
            elif p["LR schedule"] == "linear":
                scheduler = optim.lr_scheduler.PolynomialLR(
                    optimizer, total_iters=p["max epochs"], last_epoch=-1
                )
            else:
                scheduler = None

            return optimizer, scheduler

        optimizer, scheduler = create_optimizer_and_scheduler(p["optimizer"])

        model, train_time, trained_epochs, best_epoch = train_model(
            model,
            criterion,
            optimizer,
            dataloaders,
            scheduler=scheduler,
            num_epochs=p["max epochs"],
            writer=writer,
        )
        if save_models:
            torch.save(
                model.state_dict(),
                f"best_model/{exp_name}_{network_name}_{cohort}.pt",
            )
        model.eval()
        if len(training_stats_summary["validation metrics"]) == 0:
            test_cats = ["test metrics", "validation metrics"]
            if test_datasets:
                test_cats += [
                    "external test dataset: " + os.path.split(ts)[-1]
                    for ts in test_datasets
                ]
            for split in test_cats:
                tests = {
                    "acc": [],
                    "auc": [],
                    "sens": [],
                    "spec": [],
                    "F1": [],
                    "mcc": [],
                }
                training_stats_summary[split] = tests

            stats = {
                "train time": [],
                "train epochs": [],
                "best epoch": [],
                **hyperparameters,
            }
            training_stats_summary["training stats"] = stats

        stats = training_stats_summary["training stats"]
        stats["train time"].append(train_time)
        stats["train epochs"].append(trained_epochs)
        stats["best epoch"].append(best_epoch)

        for split in ["validation metrics", "test metrics"]:
            data_split = "test" if split == "test metrics" else "validation"

            test_acc, auc, sensitivity, specificity, F1, mcc = test_binary_model(
                model, dataloaders[data_split], device, dtype=DTYPE
            )

            stat_dict = training_stats_summary[split]
            stat_dict["acc"].append(round(test_acc, 4))
            stat_dict["auc"].append(round(auc, 4))
            stat_dict["sens"].append(round(sensitivity, 4))
            stat_dict["spec"].append(round(specificity, 4))
            stat_dict["F1"].append(round(F1, 4))
            stat_dict["mcc"].append(round(mcc, 4))

        print(f"{network_name} trained on {cohort}, tested on {cohort}")
        print("\nTest Accuracy:", round(test_acc, 3))
        print("Test AUC:", round(auc, 3))
        print("Sensitivity:", round(sensitivity, 3))
        print("Specificity:", round(specificity, 3))
        print("F1:", round(F1, 3))
        print("mcc:", round(mcc, 3))
        print()

        if test_datasets:
            for test_set_path, dl in external_test_dataloaders.items():
                dataset_name = os.path.split(test_set_path)[-1]
                print("\nTesting on dataset " + dataset_name)

                test_acc, auc, sensitivity, specificity, F1, mcc = test_binary_model(
                    model, dl, device, dtype=DTYPE
                )
                stat_dict = training_stats_summary[
                    "external test dataset: " + dataset_name
                ]
                stat_dict["acc"].append(round(test_acc, 4))
                stat_dict["auc"].append(round(auc, 4))
                stat_dict["sens"].append(round(sensitivity, 4))
                stat_dict["spec"].append(round(specificity, 4))
                stat_dict["F1"].append(round(F1, 4))
                stat_dict["mcc"].append(round(mcc, 4))

        all_stats[identifier] = training_stats_summary

        with open(stat_path, "w") as f:
            f.write(json.dumps(all_stats, indent=4))

        del model
        del optimizer
        torch.cuda.empty_cache()

    all_stats[identifier] = training_stats_summary
    with open(stat_path, "w") as f:
        f.write(json.dumps(all_stats, indent=4))
