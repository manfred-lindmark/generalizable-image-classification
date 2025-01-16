import os
from experiment_definitions import camelyon_design
from experiment_launcher import run_experiments
from utils import extract_demo_dataset
from stats_and_plots import json_stats_to_row_list, write_csv, plot_stats_file


def run_demo():

    # Sample pathology datasets with domain shifts:
    # CAMELYON17, part of WILDS ("A benchmark of in-the-wild distribution shifts").
    # Each dataset is from a different hospital with different model of scanner.
    if not os.path.isdir("datasets/camelyon_split_by_scanner"):
        path_of_zipped_camelyon17_subset = "demo_domain_shift_dataset.zip"
        extract_demo_dataset(path_of_zipped_camelyon17_subset)

    # Limit number of repeats and epochs (compared to article) to speed up demo:
    max_epochs = 50
    repeats = 2

    for design in camelyon_design.values():
        design["hyperparameters"][0]["repeats"] = repeats
        design["hyperparameters"][0]["max epochs"] = max_epochs

    results_file = run_experiments(
        camelyon_design, "datasets/camelyon_split_by_scanner", save_models=False
    )
    statistics_path = "statistics/" + results_file

    # A previous result can be loaded directly:
    # statistics_path = "statistics/Experiment_stats_20241114-112634.json"

    data_rows = json_stats_to_row_list(statistics_path)
    csv_file = statistics_path.replace(".json", ".csv")
    write_csv(data_rows, csv_file)
    plot_stats_file(
        csv_file,
        "Accuracy (Scanner3)",
        "Train dataset",
        "Experiment group",
    )


if __name__ == "__main__":
    run_demo()
