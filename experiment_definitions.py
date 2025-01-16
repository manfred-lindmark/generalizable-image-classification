import os


datasets_root_path = "datasets"
demo_networks = [
    "EfficientNet-v2-s"
]  # EfficientNet-v2-s recommended for best classification performance


"""
    # OPTIONS:

    datasets = [
        "ABC_test_on_DE_original", # Not cleaned or pre-processed
        "ABC_test_on_DE",          # Cleaned
        "ABC_test_on_DE_centered"  # Cleaned and centered/cropped
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    networks = [
        "MobileNet-v3-small",
        "MobileNet-v2",
        "EfficientNet-b0",
        "MobileNet-v3-large",
        "EfficientNet-b1",
        "EfficientNet-v2-s",
        "ResNet50",
        "Densenet",
        "EfficientNet-v2-m",
        "VisionTransformer-B",
        "EfficientNet-v2-l",
        "VGG16",
        "VisionTransformer-L",
    ]

    hyperparameters = {
        "repeats": 5,
        "batch size": 32,
        "gradient accumulation": 0,
        "resolution": 224,
        "max epochs": 200,
        "min epochs": 30,
        "patience": 20,
        "best model metric": "MCC",
        "class weighting": True,
        "LR restart interval": 7,
        "optimizer": "ADAMW",
        "learning rate": 1e-4,
        "momentum (SGD)": 1e-4,
        "Nesterov (SGD)": False,
        "AMSGrad (ADAMW)": True,
        "weight decay": 0.1,
        "LR schedule": "cosine",
        "dropout": "default",
        "label smoothing": 0.0,
    }

    augmentation_strategies = [
        "no augment",
        "classic",
        "multisample trivialaugment"
    ]
"""


all_networks = [
    "MobileNet-v3-small",
    "MobileNet-v2",
    "EfficientNet-b0",
    "MobileNet-v3-large",
    "EfficientNet-b1",
    "EfficientNet-v2-s",
    "ResNet50",
    "Densenet",
    "EfficientNet-v2-m",
    "VisionTransformer-B",
    "EfficientNet-v2-l",
    "VGG16",
    "VisionTransformer-L",
]

# These are used for both baseline and optimized parameter sets
fixed_parameters = {
    "repeats": 5,
    "resolution": 224,
    "max epochs": 200,
    "min epochs": 30,
    "patience": 20,
    "LR restart interval": 12,
    "momentum (SGD)": 1e-4,
    "Nesterov (SGD)": False,
    "AMSGrad (ADAMW)": True,
    "dropout": "default",
    "gradient accumulation": 0,
    "batch size": 32,  # changed if using MSTA
}

os.environ["MODEL_RESOLUTION"] = str(fixed_parameters["resolution"])

baseline_parameters = {
    "hyperparameter set": "baseline",
    "best model metric": "BCE",
    "class weighting": False,
    "optimizer": "SGD",
    "learning rate": 3e-3,
    "weight decay": 1e-4,
    "LR schedule": "linear",
    "label smoothing": 0.0,
} | fixed_parameters

optimized_parameters = {
    "hyperparameter set": "optimized",
    "best model metric": "MCC",
    "class weighting": True,
    "optimizer": "ADAMW",
    "learning rate": 1e-4,
    "weight decay": 0.1,
    "LR schedule": "cosine",
    "label smoothing": 0.1,
} | fixed_parameters


# This setup is used for the demo
camelyon_design = {
    # Baseline
    "Baseline": {
        "datasets": [
            "Scanner1",
            "Scanner2",
            "Scanner3",
        ],
        "networks": demo_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["classic"],
        "test_against_other_datasets": True,
    },
    # Only optimized augmentation
    "Optimized augmentation": {
        "datasets": [
            "Scanner1",
            "Scanner2",
            "Scanner3",
        ],
        "networks": demo_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
        "test_against_other_datasets": True,
    },
    # Only optimized hyperparameters
    "Optimized hyperparameters": {
        "datasets": [
            "Scanner1",
            "Scanner2",
            "Scanner3",
        ],
        "networks": demo_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["classic"],
        "test_against_other_datasets": True,
    },
    # Fully optimized
    "All optimized": {
        "datasets": [
            "Scanner1",
            "Scanner2",
            "Scanner3",
        ],
        "networks": demo_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
        "test_against_other_datasets": True,
    },
}


single_ds_gen_networks = [
    "ResNet50",
    "Densenet",
    "VisionTransformer-B",
    "VGG16",
]

original_datasets = [
    os.path.join("Cohorts_split", cohort) for cohort in ("A", "B", "C", "D", "E")
]

cleaned_cropped_datasets = [
    os.path.join("Cohorts_cleaned_centered_split", cohort)
    for cohort in ("A", "B", "C", "D", "E")
]

# Experiment setup for training on each dataset individually and testing against others
single_dataset_generalizability = {
    # Baseline
    "Baseline": {
        "datasets": original_datasets,
        "networks": single_ds_gen_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["no augment"],
        "test_against_other_datasets": True,
    },
    # Corrected data, baseline method
    "Optimized augmentation": {
        "datasets": cleaned_cropped_datasets,
        "networks": single_ds_gen_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["no augment"],
        "test_against_other_datasets": True,
    },
    # Optimized method, original data
    "Optimized hyperparameters": {
        "datasets": original_datasets,
        "networks": single_ds_gen_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
        "test_against_other_datasets": True,
    },
    # Fully optimized
    "All optimized": {
        "datasets": cleaned_cropped_datasets,
        "networks": single_ds_gen_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
        "test_against_other_datasets": True,
    },
}


# Design for comparing all networks
network_comparison_experiment = {
    # The best setup: cleaned and preprocessed data, optimized training, optimized augmentation
    "all_optimized": {
        "datasets": ["ABC_test_on_DE_centered"],
        "networks": all_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
    },
    # Baseline using parameters from reproduction of Habib results, original data
    "baseline": {
        "datasets": ["ABC_test_on_DE_original"],
        "networks": all_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["no augment"],
    },
    # Optimized method: best setup but without cleaned data
    "original_data": {
        "datasets": ["ABC_test_on_DE_original"],
        "networks": all_networks,
        "hyperparameters": [optimized_parameters],
        "augmentation strategies": ["multisample trivialaugment"],
    },
    # Corrected data, but baseline method
    "non_centered_data": {
        "datasets": ["ABC_test_on_DE_centered"],
        "networks": all_networks,
        "hyperparameters": [baseline_parameters],
        "augmentation strategies": ["no augment"],
    },
}


# Full factorial for 1 network: 3 data * 3 aug * 2 param * 5 repeats = 90 runs -> 60 hours
data_options = ["ABC_test_on_DE_original", "ABC_test_on_DE", "ABC_test_on_DE_centered"]
param_options = [baseline_parameters, optimized_parameters]
augmentation_options = ["no augment", "basic", "multisample trivialaugment"]
networks = ["Densenet", "EfficientNet-v2-s"]

full_factorial_design = {
    "Full factorial": {
        "datasets": data_options,
        "networks": networks,
        "hyperparameters": param_options,
        "augmentation strategies": augmentation_options,
    },
}
