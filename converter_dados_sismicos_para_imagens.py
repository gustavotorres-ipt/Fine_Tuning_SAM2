import os
import numpy as np
from PIL import Image

PASTA_INPUTS_3D = "./3d_data/inputs/"
PASTA_LABELS_3D = "./3d_data/labels/"

PASTA_INPUTS_2D = "./2d_data/inputs/"
PASTA_LABELS_2D = "./2d_data/labels/"

PASTA_INPUTS_IMAGENS = "./imagens/inputs/"
PASTA_LABELS_IMAGENS = "./imagens/labels/"

os.makedirs(PASTA_INPUTS_IMAGENS, exist_ok=True)
os.makedirs(PASTA_LABELS_IMAGENS, exist_ok=True)

def get_tipo_arquivo(nome_arquivo):
    tipo = nome_arquivo.split(".")[-1]
    return tipo

def carregar_arquivo(caminho_input: str, label=False):
    if get_tipo_arquivo(caminho_input) == "dat":
        volume_dados = np.fromfile(caminho_input, dtype=np.single)

        # Muda o dado dependendo
        if "geobodi" in caminho_input:
            volume_dados = volume_dados.reshape(1, 224, 224)

        elif "facies" in caminho_input:
            volume_dados = volume_dados.reshape(1, 768, 768)

        else:
            volume_dados = volume_dados.reshape(128, 128, 128)
        
    else: # "npy":
        volume_dados = np.load(caminho_input)

    if not(label):
        # Normalização em tons de cinza
        volume_dados = 255 * (
                volume_dados - volume_dados.min() ) / (
                volume_dados.max() - volume_dados.min()
        )
    volume_dados = volume_dados.astype(np.uint8)
    return volume_dados

def separar_volume_em_imagens(caminho_input: str, pasta_destino: str, label=False):
    nome_base_arquivo = caminho_input.split("/")[-1]
    nome_base_arquivo = ".".join(nome_base_arquivo.split(".")[:-1])

    volume_dados = carregar_arquivo(caminho_input, label)

    for i, inline_volume in enumerate(volume_dados):
        nome_arquivo_saida = f"{nome_base_arquivo}_{i}.png"

        if label:
            caminho_saida = os.path.join(PASTA_LABELS_IMAGENS, nome_arquivo_saida)
        else:
            caminho_saida = os.path.join(PASTA_INPUTS_IMAGENS, nome_arquivo_saida)

        im = Image.fromarray(inline_volume)
        # Salvar a imagem
        im.save(caminho_saida)

        print(f"{caminho_saida} salvo com sucesso.")

    # Dados 3D
    # caminho_input = os.path.join(PASTA_INPUTS_3D, arquivo)
    # separar_volume_em_imagens(caminho_input, PASTA_INPUTS_IMAGENS)

    # caminho_label = os.path.join(PASTA_LABELS_3D, arquivo)
    # separar_volume_em_imagens(caminho_label, PASTA_LABELS_IMAGENS, label=True)

arquivos = os.listdir(PASTA_INPUTS_2D)

for arquivo in arquivos:
    # Dados 2D
    caminho_input = os.path.join(PASTA_INPUTS_2D, arquivo)
    separar_volume_em_imagens(caminho_input, PASTA_INPUTS_IMAGENS)

    caminho_label = os.path.join(PASTA_LABELS_2D, arquivo)
    separar_volume_em_imagens(caminho_label, PASTA_LABELS_IMAGENS, label=True)