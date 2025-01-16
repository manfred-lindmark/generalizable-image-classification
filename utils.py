from typing import Any, Dict, Type, Union
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode as IM
from torchvision import transforms as _transforms
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType
from torchvision.transforms.v2._auto_augment import _AutoAugmentBase
from torcheval.metrics.functional import (
    binary_auroc,
    binary_f1_score,
    multiclass_f1_score,
)

from PIL import ImageOps
from math import sqrt
import warnings
import math
from torch.optim.lr_scheduler import LRScheduler


DTYPE = torch.bfloat16

RES = int(os.environ["MODEL_RESOLUTION"])


class SquarePad:
    def __call__(self, image):
        s = image.size
        if s[-1] == s[-2]:
            return image
        m = np.max(s)
        return ImageOps.pad(image, (m, m))


class RandomTranspose(nn.Module):
    random_transpose_options = (False, 2, 3, 4)

    def forward(self, img):
        choice = self.random_transpose_options[torch.randint(0, 4, (1,))]
        if not choice:
            return img
        return img.transpose(choice)


def no_augments_train(res):
    return v2.Compose(
        [
            SquarePad(),
            v2.PILToTensor(),
            v2.Resize(res, interpolation=IM.BICUBIC),
            v2.ToDtype(DTYPE, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # typically from ImageNet
        ]
    )


def classic_augments_train(res):
    return v2.Compose(
        [
            RandomTranspose(),  # 90° rotations
            SquarePad(),
            v2.PILToTensor(),
            v2.Resize(
                (int(res * 1.143), int(res * 1.143)), interpolation=IM.BICUBIC
            ),  # If model resolution is 224, this resizes to 256 (AlexNet random crop method)
            v2.RandomCrop((res, res)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(DTYPE, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # typically from ImageNet
        ]
    )


class TrivialAugmentWide(_AutoAugmentBase):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    """

    _v1_transform_cls = _transforms.TrivialAugmentWide
    _AUGMENTATION_SPACE = {
        "Identity": (lambda num_bins, height, width: None, False),
        "ShearX": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.67, num_bins),
            True,
        ),
        "ShearY": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.67, num_bins),
            True,
        ),
        "TranslateX": (
            lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins),
            True,
        ),
        "TranslateY": (
            lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins),
            True,
        ),
        "Rotate": (
            lambda num_bins, height, width: torch.linspace(0.0, 348.0, num_bins),
            True,
        ),
        "Brightness": (
            lambda num_bins, height, width: torch.linspace(0.05, 0.95, num_bins),
            True,
        ),
        "Color": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.95, num_bins),
            True,
        ),
        "Contrast": (
            lambda num_bins, height, width: torch.linspace(0.05, 0.99, num_bins),
            True,
        ),
        "Sharpness": (
            lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins),
            True,
        ),
        "Posterize": (
            lambda num_bins, height, width: (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))
            )
            .round()
            .int(),
            False,
        ),
        "Solarize": (
            lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins),
            False,
        ),
        "AutoContrast": (lambda num_bins, height, width: None, False),
        "Equalize": (lambda num_bins, height, width: None, False),
    }

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: Union[IM, int] = IM.NEAREST,
        fill: Union[_FillType, Dict[Union[Type, str], _FillType]] = None,
    ):
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        (
            flat_inputs_with_spec,
            image_or_video,
        ) = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)

        transform_id, (magnitudes_fn, signed) = self._get_random_item(
            self._AUGMENTATION_SPACE
        )

        magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
        if magnitudes is not None:
            magnitude = float(
                magnitudes[int(torch.randint(self.num_magnitude_bins, ()))]
            )
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0

        image_or_video = self._apply_image_or_video_transform(
            image_or_video,
            transform_id,
            magnitude,
            interpolation=self.interpolation,
            fill=self._fill,
        )
        return self._unflatten_and_insert_image_or_video(
            flat_inputs_with_spec, image_or_video
        )


multisample_tencrop = v2.Compose(
    [
        v2.RandomChoice(
            [
                v2.Resize(int(RES * 1.2), interpolation=IM.NEAREST),
                v2.Resize(int(RES * 1.2), interpolation=IM.BILINEAR),
                v2.Resize(int(RES * 1.2), interpolation=IM.BICUBIC),
                v2.Resize(int(RES * 1.2), interpolation=IM.LANCZOS),
            ]
        ),
        SquarePad(),
        v2.PILToTensor(),
        v2.TenCrop(RES),
    ]
)

multisample_zoom_out = v2.Compose(
    [
        RandomTranspose(),
        SquarePad(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomZoomOut(side_range=(1, 3), p=0.8),
        v2.Resize(RES),
        v2.PILToTensor(),
    ]
)


multisample_stage2 = v2.Compose(
    [
        v2.RandomRotation(180),
        TrivialAugmentWide(),
        v2.ToDtype(DTYPE, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # typically from ImageNet
    ]
)


def multisample_transform(sample):
    samples = multisample_tencrop(sample)
    samples += (multisample_zoom_out(sample), multisample_zoom_out(sample))
    samples = [multisample_stage2(s) for s in samples]
    samples = torch.stack(samples)
    return samples


def multisample_trivialaugment(res):
    return multisample_transform


def validation_transform(res):
    return v2.Compose(
        [
            SquarePad(),
            v2.PILToTensor(),
            v2.Resize(res, antialias=True, interpolation=IM.BICUBIC),
            v2.ToDtype(DTYPE, scale=True),  # to float32 in [0, 1]
            v2.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # typically from ImageNet
        ]
    )


def get_transform(augmentation_strategy, resolution):
    options = {
        "no augment": no_augments_train,
        "classic": classic_augments_train,
        "multisample trivialaugment": multisample_trivialaugment,
    }
    if augmentation_strategy not in options:
        raise ValueError
    return options[augmentation_strategy](resolution)


inv_normalize = v2.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
)


def adapt_network_classes(net_name, model, n_classes):
    if net_name in ("GoogLeNet", "ResNet18", "ResNet50", "ResNet101"):
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif net_name == "Densenet":
        model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    elif net_name[:-2] == "VisionTransformer":
        model.heads[-1] = torch.nn.Linear(model.heads[-1].in_features, n_classes)
    elif net_name[:4] == "eva_":
        model.head = torch.nn.Linear(model.head.in_features, n_classes)
    else:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, n_classes)


def binary_confusion_matrix(x, y):
    assert x.shape == y.shape and len(x.shape) == 1
    x = torch.round(x).long()
    y = y.long()
    TP = ((x + y) == 2).sum()
    TN = ((x + y) == 0).sum()
    FN = torch.logical_and(x == 0, y == 1).sum()
    FP = len(x) - TN - TP - FN
    return TP.item(), FP.item(), TN.item(), FN.item()


def matthews_correlation_coefficient(x, y):
    TP, FP, TN, FN = binary_confusion_matrix(x, y)
    if (TP * TN) - (FP * FN) == 0:
        # Not defined (division by zero)
        return 0
    else:
        return ((TP * TN) - (FP * FN)) / sqrt(
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
        )


def test_binary_model(model, test_data, device, get_auc=True, dtype=torch.bfloat16):
    output_batches = []
    label_batches = []
    model.eval()
    # iterate over test data
    with torch.inference_mode():
        for inputs, labels in test_data:
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(inputs.to(device))
                probs = F.sigmoid(logits).flatten().detach()
            output_batches.append(probs)
            label_batches.append(labels)

        x = torch.cat(output_batches)
        del output_batches
        y = torch.cat(label_batches).to(device)
        del label_batches

        if get_auc:
            auc = binary_auroc(input=x, target=y).item()
        else:
            auc = 0
        F1 = binary_f1_score(x, y).item()
        TP, FP, TN, FN = binary_confusion_matrix(x, y)
        mcc = matthews_correlation_coefficient(x, y)

        accuracy = (TP + TN) / len(x)

        print("Total test images:", len(x))
        print("Number positives (abnormal):", TP + FN)
        print("True positive predictions:", TP)
        print("Number negatives (normal):", TN + FP)
        print("True negative predictions:", TN)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

    return accuracy, auc, sensitivity, specificity, F1, mcc


def test_model(model, test_data, device, dtype=torch.bfloat16):
    output_batches = []
    label_batches = []
    model.eval()
    with torch.inference_mode():
        # iterate over test data
        for inputs, labels in test_data:
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(inputs.to(device))
            probs = F.softmax(logits, dim=-1).detach()
            output_batches.append(probs)
            label_batches.append(labels)

        x = torch.cat(output_batches)
        del output_batches
        y = torch.cat(label_batches).to(device)
        del label_batches
        _, preds = torch.max(x, 1)

        F1 = multiclass_f1_score(
            preds, y, num_classes=len(test_data.dataset.classes), average="macro"
        ).item()

        accuracy = torch.sum(preds == y).item() / len(x)

    print("Total test images:", len(x))

    return accuracy, F1, preds, y


def stats_to_tensorboard(
    writer, phase, epoch, loss, acc, F1, mcc, sensitivity, specificity, lr
):
    try:
        writer.add_scalar(
            phase + " loss",
            loss,
            epoch,
        )
        writer.add_scalar(
            phase + " accuracy",
            acc,
            epoch,
        )
        writer.add_scalar(
            phase + " F1",
            F1,
            epoch,
        )
        writer.add_scalar(
            phase + " mcc",
            mcc,
            epoch,
        )
        writer.add_scalar(
            phase + " sensitivity",
            sensitivity,
            epoch,
        )
        writer.add_scalar(
            phase + " specificity",
            specificity,
            epoch,
        )
        writer.add_scalar(
            phase + " sensitivity ÷ specificity",
            sensitivity / specificity,
            epoch,
        )
        if phase == "training":
            writer.add_scalar(
                "Learning rate",
                lr,
                epoch,
            )
    except Exception as e:
        print(e)


def extract_demo_dataset(path_to_zipped_dataset):
    if os.path.isfile(path_to_zipped_dataset):
        shutil.unpack_archive(path_to_zipped_dataset, "datasets/")
    else:
        print(
            "Check that the path of the zipfile is correct. The sample dataset should be bundled with the code."
        )


def prepare_demo_dataset():
    """Download CAMELYON17 and split"""

    # First install WILDS (pip install wilds)
    from wilds import get_dataset
    import pandas as pd
    import os
    import shutil
    import tqdm

    print(
        "Warning, this dataset is >10GB. It will take some time to download and unpack"
    )
    camelyon17 = get_dataset(dataset="camelyon17", download=True, root_dir="datasets")
    df = pd.read_csv("datasets/camelyon17_v1.0/metadata.csv", dtype=str)
    for ind in tqdm.tqdm(df.index):
        patient = df["patient"][ind]
        node = df["node"][ind]
        x = df["x_coord"][ind]
        y = df["y_coord"][ind]

        source_folder = f"camelyon17_v1.0/patches/patient_{patient}_node_{node}/"
        file = f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png"
        assert os.path.isfile(source_folder + file)

        cls = "0 normal" if df["tumor"][ind] == "0" else "1 tumor"
        dest_folder = f"camelyon_by_hospital/{df['center'][ind]}/{cls}/{file}"
        shutil.copyfile(source_folder + file, dest_folder)


class CosineAnnealingWarmRestarts(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule
    """

    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        # if T_mult < 1 or not isinstance(T_mult, int):
        #    raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = int(round(self.T_i * self.T_mult))
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
