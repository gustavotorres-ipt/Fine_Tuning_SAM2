import os
import random
import json
import numpy as np

import cv2
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO # por algum motivo precisa dar import nisso

def visualize_image(image):
    fig, ax = plt.subplots()
    color = np.array([30/255, 144/255, 255/255, 0.6])

    ax.imshow(image)
    # ax.imshow(mask_highlighted)
    plt.show()

def draw_box(img, bbox):
    point1 = (int(bbox[0]), int(bbox[1]))
    point2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(img, point1, point2, (0,255,0), 2)

def decode_segmentation_mask(segm_entry):
    encoded_mask = {'size': segm_entry['size'],
                    'counts': segm_entry['counts']}
    return pycocotools.mask.decode(encoded_mask)

def normalize_image(img):
    img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
    return img.astype('uint8')

def load_npz(folder, filename):
    img = np.load(folder + 'mask/' + filename + '.npz')
    img = cv2.cvtColor(normalize_image(img), cv2.COLOR_GRAY2BGR)
    visualize_image(img)

def test_data_from_json(folder, filename):
    img = np.load(folder + 'seismic/' + filename + '.npy')
    img = cv2.cvtColor(normalize_image(img), cv2.COLOR_GRAY2BGR)

    with open(folder + 'annotations/' + filename + '.json') as f:
        annot_list = json.load(f)

    for i,annot in enumerate(annot_list):
        bbox = annot['bbox']
        mask = decode_segmentation_mask(annot['segmentation'])
        mask = cv2.cvtColor(200*mask, cv2.COLOR_GRAY2BGR)

        blend = cv2.addWeighted(mask, 0.5, img, 0.5, 0.0)
        draw_box(blend, bbox)
        cv2.imwrite(folder + 'annotations/' + filename + f'_vis{i}.jpg', blend)

def main():
    possible_files = os.listdir("parihaka_slices/seismic")
    random.shuffle(possible_files)
    load_npz("parihaka_slices/", possible_files[0][:-4])

if __name__ == "__main__":
    main()
