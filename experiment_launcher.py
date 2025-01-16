from utils import get_transform
from train import run_training
from datetime import datetime
import numpy as np
import os


GLOBAL_SEED = 0

# Can be used to start at a later stage if a previous run was interrupted before finishing
CONTINUE_FROM_COMBINATION = 0


def count_exp_design_combinations(exp_design):
    num_combinations = int(
        np.prod(
            [
                len(x)
                for x in [
                    exp_design["datasets"],
                    exp_design["networks"],
                    exp_design["hyperparameters"],
                    exp_design["augmentation strategies"],
                ]
            ]
        )
    )
    return num_combinations


def run_experiments(experimental_designs, datasets_root_path, save_models=True):

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    statistics_filename = f"Experiment_stats_{date_time}.json"

    combination_id = 1
    num_combinations = sum(
        count_exp_design_combinations(design)
        for design in experimental_designs.values()
    )

    for exp_name, exp in experimental_designs.items():

        datasets = exp["datasets"]
        networks = exp["networks"]
        hyperparameter_sets = exp["hyperparameters"]
        augmentations = exp["augmentation strategies"]

        repeats = hyperparameter_sets[0]["repeats"]
        print(f"\nTotal number of combinations: {num_combinations}")
        print(
            f"With {repeats} repeats the total number of training runs is {num_combinations*repeats}."
        )

        for net_name in networks:
            for augmentation_strategy in augmentations:
                for params in hyperparameter_sets:
                    params["augmentation"] = augmentation_strategy
                    for dataset_name in datasets:
                        params["combination id"] = combination_id
                        if exp["test_against_other_datasets"]:
                            external_test_datasets = [
                                os.path.join(datasets_root_path, ds)
                                for ds in datasets
                                if ds != dataset_name
                            ]
                        else:
                            external_test_datasets = False
                        if combination_id < CONTINUE_FROM_COMBINATION:
                            print(f"Skipping combination {combination_id}")
                        else:
                            print(f"\n\nStarting combination {combination_id}.\n")
                            if augmentation_strategy == "multisample trivialaugment":
                                params["batch size"] = 8
                            else:
                                params["batch size"] = 32

                            run_training(
                                exp_name,
                                os.path.join(datasets_root_path, dataset_name),
                                net_name,
                                params,
                                get_transform(
                                    augmentation_strategy, params["resolution"]
                                ),
                                GLOBAL_SEED,
                                statistics_filename,
                                test_datasets=external_test_datasets,
                                save_models=save_models,
                            )
                            print(
                                f"Finished combination {combination_id} out of {num_combinations}"
                            )
                        combination_id += 1
    return statistics_filename


if __name__ == "__main__":
    from experiment_definitions import camelyon_design, datasets_root_path

    run_experiments(camelyon_design, datasets_root_path, save_models=True)
