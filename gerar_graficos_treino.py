import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TRAINING_FILES = {
    'tiny'     : 'resultados_sam2.1_hiera_tiny_seismic.json',
    'small'    : 'resultados_sam2.1_hiera_small_seismic.json',
    'base_plus': 'resultados_sam2.1_hiera_base_plus_seismic.json',
    'large'    : 'resultados_sam2.1_hiera_large_seismic.json'
}

def load_training_file(model_size: str) -> dict[str, list]:
    with open(TRAINING_FILES[model_size]) as f:
        dict_training = json.load(f)
        return dict_training

def plot_results(dict_training: dict, metric: str, model_size: str) -> None:
    plt.tight_layout()
    sns.set_style("darkgrid")

    if metric == "iou":
        data = {
            "Treino": dict_training["iou_train"],
            "Validação": dict_training["iou_val"]
        }
        lineplot = sns.lineplot(data=data, legend='full', palette="deep")
        plt.title(f"Intersection Over Union ({model_size})")
        plt.ylim(0.5, 1)
    else:
        data = {
            "Treino": dict_training["loss_train"],
            "Validação": dict_training["loss_val"]
        }
        lineplot = sns.lineplot(data=data, legend='full', palette="deep")
        plt.title(f"Loss ({model_size})")
        plt.ylim(0, 0.06)

    lineplot.set(xlabel='Épocas de Treinamento', ylabel=metric.title())

    arquivo_output = f"{model_size}_{metric}.png"

    fig = lineplot.get_figure()
    fig.savefig(f"graficos_treinamento/{arquivo_output}")
    print(f"{arquivo_output} salvo com sucesso.")
    plt.clf() 

if __name__ == '__main__':
    for model_size in TRAINING_FILES:
        dict_training = load_training_file(model_size)

        plot_results(dict_training, "iou", model_size.title())
        plot_results(dict_training, "loss", model_size.title())
