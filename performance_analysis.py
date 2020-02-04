import os
from matplotlib import pyplot as plt
import numpy as np


results = {}
for path, subfolders, files in os.walk("Analyse/Analyse_total_de_la_sequence_video"):
    print(path)
    folder_name = path.split("\\")[-1].split("/")[-1]
    for file in files:
        if not file.endswith(".txt"):
            continue
        result_file = open(path + "/" + file)
        line = result_file.readline().lstrip()
        print(file, line)
        values = [float(value) for value in line.split(" ")]
        if file not in results:
            results[file] = {}
        results[file][folder_name] = values

print(results)

dataset_groups = {
    "baseline": [],
    "dynamicBackground": [],
    "turbulence": []
}

for i, key in enumerate(list(results.keys())):
    split_key = key.split("_")
    for part in split_key:
        for group in list(dataset_groups.keys()):
            if part == group:
                dataset_groups[group].append(i)
                break

labels = [dataset.split(".")[0].split("result_")[-1] for dataset in list(results.keys())]
values_classification = [values['Efficacite_classification'] for values in list(results.values())]
values_classification_filtered = [values['Efficacite_classification_filtre'] for values in list(results.values())]
values_flow_noisy = [values['Efficacite_flow_avec_du_bruit'] for values in list(results.values())]
values_flow_blurry = [values['Efficacite_flow_pas_de bruit'] for values in list(results.values())]
values_flow_blurry_cropped = [values['Efficacite_flow_pas_de bruit_réduction_objet'] for values in list(results.values())]

bar_width = 0.35


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        value = round(height * 100) / 100
        ax.annotate('{}'.format(value),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', size=7)


for i, metric in enumerate(["Précision", "Rappel"]):
    # Graph 1: separated for each dataset
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - bar_width, np.array(values_classification)[:, i], bar_width/2, label='Faster R-CNN')
    rects2 = ax.bar(x - bar_width/2, np.array(values_classification_filtered)[:, i], bar_width / 2, label='Faster R-CNN filtré')
    rects3 = ax.bar(x, np.array(values_flow_noisy)[:, i], bar_width/2, label='Flux optique fenêtre 15px')
    rects4 = ax.bar(x + bar_width/2, np.array(values_flow_blurry)[:, i], bar_width/2, label='Flux optique fenêtre 35px')
    rects5 = ax.bar(x + bar_width, np.array(values_flow_blurry_cropped)[:, i], bar_width/2, label='Flux optique fenêtre 35px réduit')

    ax.set_xlabel('Bases de données')
    ax.set_ylabel('Scores')
    ax.set_title(metric + " (pour chaque base de données)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(10)
    ax.legend()

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()

    # Graph 2: averaged by dataset group
    x = np.arange(len(dataset_groups))
    fig, ax = plt.subplots()

    faster_rcnn_values_by_group = [np.array(values_classification)[dataset_groups[group], i].mean() for group in dataset_groups]
    faster_rcnn_filtre_values_by_group = [np.array(values_classification_filtered)[dataset_groups[group], i].mean() for group in dataset_groups]
    optical_flow_15px_values_by_group = [np.array(values_flow_noisy)[dataset_groups[group], i].mean() for group in dataset_groups]
    optical_flow_35px_values_by_group = [np.array(values_flow_blurry)[dataset_groups[group], i].mean() for group in dataset_groups]
    optical_flow_35px_cropped_values_by_group = [np.array(values_flow_blurry_cropped)[dataset_groups[group], i].mean() for group in dataset_groups]
    rects1 = ax.bar(x - bar_width, faster_rcnn_values_by_group, bar_width / 2, label='Faster R-CNN')
    rects2 = ax.bar(x - bar_width / 2, faster_rcnn_filtre_values_by_group, bar_width / 2, label='Faster R-CNN filtré')
    rects3 = ax.bar(x, optical_flow_15px_values_by_group, bar_width / 2, label='Flux optique fenêtre 15px')
    rects4 = ax.bar(x + bar_width / 2, optical_flow_35px_values_by_group, bar_width / 2, label='Flux optique fenêtre 35px')
    rects5 = ax.bar(x + bar_width, optical_flow_35px_cropped_values_by_group, bar_width / 2, label='Flux optique fenêtre 35px réduit')

    ax.set_xlabel('Groupes de base de données')
    ax.set_ylabel('Scores')
    ax.set_title(metric + " (moyenne par groupe de base de données)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(dataset_groups.keys()))
    ax.legend()

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()

    # Graph 3: averaged on all datasets
    bar_values = [
        np.array(values_classification)[:, i].mean(),
        np.array(values_classification_filtered)[:, i].mean(),
        np.array(values_flow_noisy)[:, i].mean(),
        np.array(values_flow_blurry)[:, i].mean(),
        np.array(values_flow_blurry_cropped)[:, i].mean()
    ]
    bar_labels = [
        'Faster R-CNN',
        'Faster R-CNN filtré',
        'Flux optique fenêtre 15px',
        'Flux optique fenêtre 35px',
        'Flux optique fenêtre 35px réduit'
    ]
    x = np.arange(len(bar_values))
    plt.bar(x, bar_values, align='center')
    plt.xticks(x, bar_labels)
    plt.grid(axis='y')
    plt.title(metric + " (moyenne sur toutes les bases de données)")
    plt.xlabel('Méthodes')
    plt.ylabel('Scores')

    plt.show()
