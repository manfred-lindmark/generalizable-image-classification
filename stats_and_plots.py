import json
import csv
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

colors = ["#EE7733", "#33BBEE", "#EE3377", "#009988", "#CC3311", "#0077BB"]
darkcolors = [sns.set_hls_values(color=c, h=None, l=None, s=0.4) for c in colors]
sns.set_theme(font="Arial", style="whitegrid", palette=colors)

base_stat_variables = [
    "Experiment group",
    "Network",
    "Train dataset",
    "Augmentation strategy",
    "Training hyperparameters",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "AUROC",
    "F1-score",
    "MCC",
    "Accuracy (val)",
    "Sensitivity (val)",
    "Specificity (val)",
    "AUROC (val)",
    "F1-score (val)",
    "MCC (val)",
]

key_name_map = {
    "acc": "Accuracy",
    "sens": "Sensitivity",
    "spec": "Specificity",
    "auc": "AUROC",
    "F1": "F1-score",
    "mcc": "MCC",
}


def plot_stats_file(csv_file_path, y_variable, x_variable, color_by_variable):
    df = pandas.read_csv(csv_file_path)
    df.rename(columns={col: col.replace("_", " ").capitalize() for col in df.columns})

    ax = sns.barplot(
        x=x_variable,
        y=y_variable,
        hue=color_by_variable,
        data=df,
        errorbar="sd",
        palette=colors[:4],
        err_kws={"linewidth": 1, "color": "black", "zorder": 100},
        capsize=0.2,
        saturation=1,
        legend=True,
    )

    ax.set(
        ylabel=y_variable,
    )

    sns.stripplot(
        x=x_variable,
        y=y_variable,
        hue=color_by_variable,
        palette=darkcolors[:4],
        data=df,
        s=5,
        dodge=True,
        alpha=0.9,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        jitter=0.0,  # 35
        legend=False,
    )

    # Necessary to rotate labels to fit many long names in a small plot...
    # plt.setp(
    #     ax.xaxis.get_majorticklabels(), rotation=30, ha="right", rotation_mode="anchor"
    # )

    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(
        csv_file_path.replace(".csv", f" {x_variable} {y_variable}.png"), dpi=144
    )
    plt.show()


def write_csv(csv_rows, dest_path):
    with open(dest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, dialect="excel")
        writer.writerows(csv_rows)
        print(f"Wrote {str(len(csv_rows))} rows to csv file.")


def json_stats_to_row_list(filepath):

    with open(filepath) as f:
        stat_dict = json.load(f)

    external_tests = set()
    for group in stat_dict:
        external_tests = external_tests.union(
            {key for key in stat_dict[group] if "external test dataset" in key}
        )
    external_tests = sorted(list(external_tests))
    external_test_datasets = [
        t.replace("external test dataset: ", "") for t in external_tests
    ]
    stat_variables = base_stat_variables.copy()
    for ds in external_test_datasets:
        stat_variables += [
            f"Accuracy ({ds})",
            f"Sensitivity ({ds})",
            f"Specificity ({ds})",
            f"AUROC ({ds})",
            f"F1-score ({ds})",
            f"MCC ({ds})",
        ]

    rows = [stat_variables]  # Column headers
    for group in stat_dict:
        repeats = stat_dict[group]["training stats"]["repeats"]
        net = stat_dict[group]["network"]
        train_dataset = stat_dict[group]["dataset"]
        aug = stat_dict[group]["training stats"]["augmentation"]
        params = stat_dict[group]["hyperparameter set"]

        for i in range(repeats):
            row = [group.split(" 202")[0], net, train_dataset, aug, params]
            for metrics in ["test metrics", "validation metrics"]:
                metric_dict = stat_dict[group][metrics]
                row += [
                    metric_dict["acc"][i],
                    metric_dict["sens"][i],
                    metric_dict["spec"][i],
                    metric_dict["auc"][i],
                    metric_dict["F1"][i],
                    metric_dict["mcc"][i],
                ]

            row += [""] * (len(stat_variables) - len(row))
            for key, stat in key_name_map.items():
                for ds in external_test_datasets:
                    if f"external test dataset: {ds}" in stat_dict[group]:
                        metric_dict = stat_dict[group][f"external test dataset: {ds}"]
                        column = stat_variables.index(f"{stat} ({ds})")
                        row[column] = metric_dict[key][i]
                    elif ds == train_dataset:
                        row[stat_variables.index(f"{stat} ({ds})")] = stat_dict[group][
                            "test metrics"
                        ][key][i]

            rows.append(row)

    return rows
