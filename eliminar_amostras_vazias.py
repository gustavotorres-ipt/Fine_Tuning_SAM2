import os
import cv2 # type: ignore
import numpy as np

imagens_dir = "./imagens"

total = 0

for _, nome_arquivo in enumerate(os.listdir(f"{imagens_dir}/inputs")):  # go over all folder annotation
    path_imagem = f"{imagens_dir}/inputs/{nome_arquivo}"
    path_label  = f"{imagens_dir}/labels/{nome_arquivo}"

    Img = cv2.imread(path_imagem)  # read image
    ann_map = cv2.imread(path_label)

    if np.unique(ann_map).shape[0] == 1:
        os.remove(path_imagem)
        os.remove(path_label)
    print(path_imagem)